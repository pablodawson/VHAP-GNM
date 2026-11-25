# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


from vhap_gnm.config.base import import_module, PhotometricStageConfig, BaseTrackingConfig
from vhap_gnm.model.gnm import GnmHead
from gnm.shape.gnm_common import axis_angle_to_rotation_matrix as axis_angle_to_matrix
from vhap_gnm.util.mesh import (
    get_mtl_content,
    get_obj_content,
    normalize_image_points,
)
from vhap_gnm.util.log import get_logger
from vhap_gnm.util.visualization import plot_landmarks_2d

from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
from matplotlib import cm
from typing import Literal
from functools import partial
import tyro
import yaml
from datetime import datetime
import threading
from typing import Optional
from collections import defaultdict
from copy import deepcopy
import time
import os


class GnmTracker:
    def __init__(self, cfg: BaseTrackingConfig):
        self.cfg = cfg

        self.device = cfg.device
        self.tb_writer = None

        # model
        self.head = GnmHead(
            face_clusters=cfg.model.tex_clusters,
            couple_inner_offsets=cfg.model.couple_inner_offsets,
            ).to(self.device)

        # renderer for visualization, dense photometric energy
        if self.cfg.render.backend == 'nvdiffrast':
            from vhap_gnm.util.render_nvdiffrast import NVDiffRenderer

            self.render = NVDiffRenderer(
                use_opengl=self.cfg.render.use_opengl,
                lighting_type=self.cfg.render.lighting_type,
                lighting_space=self.cfg.render.lighting_space,
                disturb_rate_fg=self.cfg.render.disturb_rate_fg,
                disturb_rate_bg=self.cfg.render.disturb_rate_bg,
                fid2cid=self.head.mask.fid2cid,
            )
        else:
            raise NotImplementedError(f"Unknown renderer backend: {self.cfg.render.backend}")

        if self.cfg.w.normal_regions:
            vids = self.head.mask.get_vid_by_region(list(self.cfg.w.normal_regions))
            normal_region_vertex_mask = torch.zeros(self.head.mask.num_verts, dtype=torch.bool, device=self.device)
            normal_region_vertex_mask[vids] = True
            self.normal_region_vertex_mask = normal_region_vertex_mask

    def load_from_tracked_gnm_params(self, fp):
        """
        loads checkpoint from tracked_gnm_params file. Counterpart to save_result()
        :param fp:
        :return:
        """
        report = np.load(fp)

        # LOADING PARAMETERS
        def load_param(param, ckpt_array):
            param.data[:] = torch.from_numpy(ckpt_array).to(param.device)

        def load_param_list(param_list, ckpt_array):
            for i in range(min(len(param_list), len(ckpt_array))):
                load_param(param_list[i], ckpt_array[i])

        load_param_list(self.rotation, report["rotation"])
        load_param_list(self.translation, report["translation"])
        load_param_list(self.head_pose, report["head_pose"])
        load_param_list(self.eyes_pose, report["eyes_pose"])
        load_param(self.identity, report["identity"])
        load_param_list(self.expr, report["expr"])
        load_param(self.lights, report["lights"])
        # self.frame_idx = report["n_processed_frames"]
        if not self.calibrated:
            load_param(self.focal_length, report["focal_length"])

        if "tex" in report:
            load_param(self.tex, report["tex"])
        else:
            self.logger.warn("No tex found in gnm_params!")

        if self.cfg.model.use_static_offset:
            if "static_offset" in report:
                load_param(self.static_offset, report["static_offset"])
            else:
                self.logger.warn("No static_offset found in gnm_params!")

        if self.cfg.model.use_dynamic_offset:
            if "dynamic_offset" in report:
                load_param_list(self.dynamic_offset, report["dynamic_offset"])
            else:
                self.logger.warn("No dynamic_offset found in gnm_params!")

    def trimmed_decays(self, is_init):
        decays = {}
        for k, v in self.decays.items():
            if is_init and "init" in k or not is_init and "init" not in k:
                decays[k.replace("_init", "")] = v
        return decays

    def clear_cache(self):
        self.render.clear_cache()

    def fill_cam_params_into_sample(self, sample):
        """
        Adds intrinsics and extrinics to sample, if data is not calibrated
        """
        if self.calibrated:
            assert "intrinsic" in sample
            assert "extrinsic" in sample
        else:
            b, _, h, w = sample["rgb"].shape
            # K = torch.eye(3, 3).to(self.device)

            # denormalize cam params
            f = self.focal_length * max(h, w)
            cx, cy = torch.tensor([[0.5*w], [0.5*h]]).to(f)

            sample["intrinsic"] = torch.stack([f, f, cx, cy], dim=1)
            sample["extrinsic"] = self.RT[None, ...].expand(b, -1, -1)

    def configure_optimizer(self, params, lr_scale=1.0):
        """
        Creates optimizer for the given set of parameters
        :param params:
        :return:
        """
        # copy dict because we will call 'pop'
        params = params.copy()
        param_groups = []
        default_lr = self.cfg.lr.base

        # dict map group name to param dict keys
        group_def = {
            "translation": ["translation"],
            "expr": ["expr"],
            "light": ["lights"],
        }
        if not self.calibrated:
            group_def ["cam"] = ["cam"]
        if self.cfg.model.use_static_offset:
            group_def ["static_offset"] = ["static_offset"]
        if self.cfg.model.use_dynamic_offset:
            group_def ["dynamic_offset"] = ["dynamic_offset"]

        # dict map group name to lr
        group_lr = {
            "translation": self.cfg.lr.translation,
            "expr": self.cfg.lr.expr,
            "light": self.cfg.lr.light,
        }
        if not self.calibrated:
            group_lr["cam"] = self.cfg.lr.camera
        if self.cfg.model.use_static_offset:
            group_lr["static_offset"] = self.cfg.lr.static_offset
        if self.cfg.model.use_dynamic_offset:
            group_lr["dynamic_offset"] = self.cfg.lr.dynamic_offset

        for group_name, param_keys in group_def.items():
            selected = []
            for p in param_keys:
                if p in params:
                    selected += params.pop(p)
            if len(selected) > 0:
                param_groups.append({"params": selected, "lr": group_lr[group_name] * lr_scale})

        # create default group with remaining params
        selected = []
        for _, v in params.items():
            selected += v
        param_groups.append({"params": selected})

        optim = torch.optim.Adam(param_groups, lr=default_lr * lr_scale)
        return optim

    def forward_head(self, timesteps):
        """
        Evaluates the GNM head model using the given parameters
        :param timesteps:
        :return:
        """
        dynamic_offset = self.dynamic_offset[timesteps] if self.cfg.model.use_dynamic_offset else None

        ret = self.head(
            self.identity[None, ...].expand(len(timesteps), -1),
            self.expr[timesteps],
            self.rotation[timesteps],
            self.head_pose[timesteps],
            self.eyes_pose[timesteps],
            self.translation[timesteps],
            return_verts_cano=True,
            static_offset=self.static_offset,
            dynamic_offset=dynamic_offset,
        )
        verts, verts_cano, lmks = ret[0], ret[1], ret[2]
        albedos = self.get_albedo().expand(len(timesteps), -1, -1, -1)
        return verts, verts_cano, lmks, albedos

    def get_albedo(self):
        return self.tex[None, ...]

    @torch.no_grad()
    def init_texture_from_view(self, sample):
        """
        Warm-starts the optimizable texture by baking the visible facial
        appearance from the first frame, using the rigid pose + identity fit
        from lmk_init_all. This lets the photometric stages spend their budget
        refining geometry from the start, instead of also having to learn the
        entire facial appearance from scratch starting from a flat gray texture.
        """
        self.fill_cam_params_into_sample(sample)
        verts, _, _, _ = self.forward_head(sample["timestep_index"])

        verts_uv = self.head.verts_uvs.clone()
        verts_uv[:, 1] = 1 - verts_uv[:, 1]

        baked, coverage = self.render.bake_texture_from_view(
            verts[[0]], self.head.faces, verts_uv, self.head.textures_idx,
            sample["extrinsic"][[0]].to(self.device), sample["intrinsic"][[0]].to(self.device),
            sample["rgb"][[0]].to(self.device), self.cfg.model.tex_resolution,
        )
        coverage_ratio = coverage.float().mean().item()
        if coverage_ratio < 0.05:
            self.logger.warning(
                f"Texture bake only covered {coverage_ratio:.1%} of the map; keeping the gray initialization."
            )
            return
        self.tex.data.copy_(baked)
        self.logger.info(f"Initialized texture from frame 0's view ({coverage_ratio:.1%} coverage).")

    def rasterize_head(
        self, sample, verts, faces, camera_index=None, train_mode=False
    ):
        """
        Rasterizes the GNM head mesh
        :param verts:
        :param albedos:
        :param K:
        :param RT:
        :param resolution:
        :param use_cache:
        :return:
        """
        # cameras parameters
        K = sample["intrinsic"].clone().to(self.device)
        RT = sample["extrinsic"].to(self.device)
        if camera_index is not None:
            K = K[[camera_index]]
            RT = RT[[camera_index]]

        H, W = self.image_size
        image_size = H, W
        
        # rasterize fragments
        rast_dict = self.render.rasterize(verts, faces, RT, K, image_size, False, train_mode)
        return rast_dict

    @torch.no_grad()
    def get_background_color(self, gt_rgb, gt_alpha, stage):
        if stage is None:  # when stage is None, it means we are in the evaluation mode
            background = self.cfg.render.background_eval
        else:
            background = self.cfg.render.background_train

        if background == 'target':
            """use gt_rgb as background"""
            color = gt_rgb.permute(0, 2, 3, 1)
        elif background == 'white':
            color = [1, 1, 1]
        elif background == 'black':
            color = [0, 0, 0]
        else:
            raise NotImplementedError(f"Unknown background mode: {background}")
        return color
    
    def render_rgba(
            self, rast_dict, verts, faces, albedos, lights, background_color=[1, 1, 1],
            align_texture_except_fid=None, align_boundary_except_vid=None, enable_disturbance=False,
        ):
        """
        Renders the rgba image from the rasterization result and
        the optimized texture + lights
        """
        faces_uv = self.head.textures_idx
        if self.cfg.render.backend == 'nvdiffrast':
            verts_uv = self.head.verts_uvs.clone()
            verts_uv[:, 1] = 1 - verts_uv[:, 1]
            tex = albedos

            render_out = self.render.render_rgba(
                rast_dict, verts, faces, verts_uv, faces_uv, tex, lights, background_color,
                align_texture_except_fid, align_boundary_except_vid, enable_disturbance
            )
            render_out = {k: v.permute(0, 3, 1, 2) for k, v in render_out.items()}
        else:
            raise NotImplementedError(f"Unknown renderer backend: {self.cfg.render.backend}")

        return render_out

    def compute_lmk_energy(self, sample, pred_lmks, disable_jawline_landmarks=False):
        """
        Computes the landmark energy loss term between groundtruth landmarks and GNM landmarks
        :param sample:
        :param pred_lmks:
        :return: the lmk loss for all 68 facial landmarks, a separate 2 pupil landmark loss and
                 a relative eye close term
        """
        img_size = sample["rgb"].shape[-2:]

        # ground-truth landmark
        lmk2d = sample["lmk2d"].clone().to(pred_lmks)
        lmk2d, confidence = lmk2d[:, :, :2], lmk2d[:, :, 2]
        lmk2d[:, :, 0], lmk2d[:, :, 1] = normalize_image_points(
            lmk2d[:, :, 0], lmk2d[:, :, 1], img_size
        )

        # predicted landmark
        K = sample["intrinsic"].to(self.device)
        RT = sample["extrinsic"].to(self.device)
        pred_lmk_ndc = self.render.world_to_ndc(pred_lmks, RT, K, img_size, flip_y=True)
        pred_lmk2d = pred_lmk_ndc[:, :, :2]

        if not self.cfg.w.always_enable_jawline_landmarks and disable_jawline_landmarks:
            diff = lmk2d[:, 17:68] - pred_lmk2d[:, 17:68]
            confidence = confidence[:, 17:68]
        else:
            diff = lmk2d[:, :68] - pred_lmk2d[:, :68]
            confidence = confidence[:, :68]

            # increase weight for nose landmarks since they are usually robust
            # https://ibug.doc.ic.ac.uk/media/uploads/images/300-w/figure_1_68.jpg
            confidence[:, 27:36] *= 10

        # compute general landmark term
        lmk_loss = torch.norm(diff, dim=2, p=1) * confidence

        result_dict = {
            "gt_lmk2d": lmk2d,
            "pred_lmk2d": pred_lmk2d,
        }

        return lmk_loss.mean(), result_dict

    def compute_photometric_energy(
        self,
        sample,
        verts,
        faces,
        albedos,
        rast_dict,
        step_i=None,
        stage=None,
    ):
        """
        Computes the dense photometric energy
        :param sample:
        :param vertices:
        :param albedos:
        :return:
        """
        gt_rgb = sample["rgb"].to(verts)
        if "alpha" in sample:
            gt_alpha = sample["alpha_map"].to(verts)
        else:
            gt_alpha = None

        lights = self.lights[None] if self.lights is not None else None
        bg_color = self.get_background_color(gt_rgb, gt_alpha, stage)

        align_texture_except_fid = self.head.mask.get_fid_by_region(
            self.cfg.pipeline[stage].align_texture_except
        ) if stage is not None else None
        align_boundary_except_vid = self.head.mask.get_vid_by_region(
            self.cfg.pipeline[stage].align_boundary_except
        ) if stage is not None else None

        render_out = self.render_rgba(
            rast_dict, verts, faces, albedos, lights, bg_color, 
            align_texture_except_fid, align_boundary_except_vid,
            enable_disturbance=stage!=None,
        )

        pred_rgb = render_out['rgba'][:, :3]
        pred_alpha = render_out['rgba'][:, 3:]
        pred_mask = render_out['rgba'][:, [3]].detach() > 0
        pred_mask = pred_mask.expand(-1, 3, -1, -1)

        results_dict = render_out

        # ---- rgb loss ----
        error_rgb = gt_rgb - pred_rgb
        color_loss = error_rgb.abs().sum() / pred_mask.detach().sum()

        results_dict.update(
            {
                "gt_rgb": gt_rgb,
                "pred_rgb": pred_rgb,
                "error_rgb": error_rgb,
                "pred_alpha": pred_alpha,
            }
        )

        # ---- silhouette loss ----
        # error_alpha = gt_alpha - pred_alpha
        # mask_loss = error_alpha.abs().sum()

        # results_dict.update(
        #     {
        #         "gt_alpha": gt_alpha,
        #         "error_alpha": error_alpha,
        #     }
        # )

        # ---- background loss ----
        # bg_mask = gt_alpha < 0.5
        # error_alpha = gt_alpha - pred_alpha
        # error_alpha = torch.where(bg_mask, error_alpha, torch.zeros_like(error_alpha))
        # mask_loss = error_alpha.abs().sum() / bg_mask.sum()

        # results_dict.update(
        #     {
        #         "gt_alpha": gt_alpha,
        #         "error_alpha": error_alpha,
        #     }
        # )

        # --------
        # photo_loss = color_loss + mask_loss
        photo_loss = color_loss
        # photo_loss = mask_loss
        return photo_loss, results_dict

    def compute_normal_energy(self, sample, verts, faces, rast_dict):
        """
        Computes a camera-space normal-supervision energy against a Sapiens2
        prediction, cached per frame by vhap_gnm.util.normal_detector. Unlike
        landmarks (68 sparse points that don't cover e.g. the cheeks) or
        photometric residual (which alone is prone to explaining appearance
        with implausible geometry under monocular depth ambiguity), this
        gives a dense, independent geometric constraint everywhere the
        estimator is confident.
        """
        gt_normal = sample["normal"].to(verts)  # (B, 3, H, W), unit vectors, top-down/camera-space
        gt_normal = gt_normal.permute(0, 2, 3, 1)  # (B, H, W, 3)

        RT = sample["extrinsic"].to(verts)
        pred_normal, fg_mask = self.render.render_normal_camera_space(rast_dict, verts, faces, RT)

        # Mask by the *rendered* mesh's own coverage (fg_mask) AND the actual
        # ground-truth person segmentation (alpha_map, independent of the current,
        # possibly still-wrong geometry fit). Without the latter, a coarse mesh that
        # currently projects onto background/clothing gets compared against whatever
        # normal Sapiens predicted there, which can drag geometry outside the
        # person's real bounds instead of just refining shape within them.
        mask = fg_mask[..., 0] & (gt_normal.norm(dim=-1) > 1e-6)
        if "alpha_map" in sample:
            gt_alpha = sample["alpha_map"].to(verts)  # (B, 1, H, W), in [0, 1]
            mask = mask & (gt_alpha[:, 0] > 0.5)
        if self.cfg.w.normal_regions:
            # GNM is head-only: regions like "neck" are a crude stump that
            # can't match the real subject's neck/shoulders/body, so we only
            # trust normal supervision on the region set below (default:
            # hair + face) rather than the whole rendered silhouette.
            region_mask = self.render.render_vertex_region_mask(
                rast_dict, faces, self.normal_region_vertex_mask
            )
            mask = mask & region_mask[..., 0]
        if mask.sum() == 0:
            return torch.zeros((), device=verts.device), {}

        cos_sim = (pred_normal * gt_normal).sum(dim=-1)
        normal_loss = (1 - cos_sim)[mask].mean()

        return normal_loss, {"pred_normal": pred_normal.permute(0, 3, 1, 2), "gt_normal": gt_normal.permute(0, 3, 1, 2)}

    def compute_regularization_energy(self, result_dict, verts, verts_cano, lmks, albedos, timesteps, stage):
        """
        Computes the energy term that penalizes strong deviations from the GNM base model
        """
        log_dict = {}

        std_expr = 1
        std_identity = 1

        # pose smoothness term
        if self.opt_dict['pose'] and 'tracking' in stage:
            E_pose_smooth = self.compute_pose_smooth_energy(timesteps)
            log_dict["smooth_pose"] = E_pose_smooth

        # joint regularization term
        if self.opt_dict['joints']:
            reg_joint = self.compute_joint_L2_energy(timesteps)
            log_dict["reg_joint"] = reg_joint
            if 'tracking' in stage:
                joint_smooth = self.compute_joint_smooth_energy(timesteps)
                log_dict["smooth_joint"] = joint_smooth

        # expression regularization
        if self.opt_dict['expr']:
            reg_expr = (self.expr[timesteps] / std_expr) ** 2
            log_dict["reg_expr"] = self.cfg.w.reg_expr * reg_expr.mean()
            if 'tracking' in stage:
                expr_smooth = self.compute_expr_smooth_energy(timesteps)
                log_dict["smooth_expr"] = expr_smooth

        # identity regularization
        if self.opt_dict['identity']:
            reg_identity = (self.identity / std_identity) ** 2
            log_dict["reg_identity"] = self.cfg.w.reg_identity * reg_identity.mean()

        # texture regularization
        if self.opt_dict['texture']:
            if self.cfg.w.reg_tex_tv is not None:
                tex = self.get_albedo()[0]  # (3, H, W)
                tv_y = (tex[..., :-1, :] - tex[..., 1:, :]) ** 2
                tv_x = (tex[..., :, :-1] - tex[..., :, 1:]) ** 2
                tv = tv_y.reshape(tv_y.shape[0], -1) + tv_x.reshape(tv_x.shape[0], -1)
                w_reg_tex_tv = self.cfg.w.reg_tex_tv * self.cfg.data.scale_factor ** 2
                if self.cfg.data.n_downsample_rgb is not None:
                    w_reg_tex_tv /= (self.cfg.data.n_downsample_rgb ** 2)
                log_dict["reg_tex_tv"] = w_reg_tex_tv * tv.mean()

        # lighting parameters regularization
        if self.opt_dict['lights']:
            if self.cfg.w.reg_light is not None and self.lights is not None:
                reg_light = (self.lights - self.lights_uniform) ** 2
                log_dict["reg_light"] = self.cfg.w.reg_light * reg_light.mean()
            
            if self.cfg.w.reg_diffuse is not None and self.lights is not None:
                diffuse = result_dict['diffuse_detach_normal']
                reg_diffuse = F.relu(diffuse.max() - 1) + diffuse.var(dim=1).mean()
                log_dict["reg_diffuse"] = self.cfg.w.reg_diffuse * reg_diffuse
            
        # offset regularization
        if self.opt_dict['static_offset'] or self.opt_dict['dynamic_offset']:
            if self.static_offset is not None or self.dynamic_offset is not None:
                offset = 0
                if self.static_offset is not None:
                    offset += self.static_offset
                if self.dynamic_offset is not None:
                    offset += self.dynamic_offset[timesteps]

                if self.cfg.w.reg_offset_lap is not None:
                    # laplacian loss
                    vert_wo_offset = (verts_cano - offset).detach()
                    reg_offset_lap = self.compute_laplacian_smoothing_loss(
                        vert_wo_offset, vert_wo_offset + offset
                    )
                    if len(self.cfg.w.reg_offset_lap_relax_for) > 0:
                        w = self.scale_vertex_weights_by_region(
                            weights=torch.ones_like(verts[:, :, :1]),
                            scale_factor=self.cfg.w.reg_offset_lap_relax_coef,
                            region=self.cfg.w.reg_offset_lap_relax_for,
                        )
                        reg_offset_lap *= w
                    log_dict["reg_offset_lap"] = self.cfg.w.reg_offset_lap * reg_offset_lap.mean()

                if self.cfg.w.reg_offset is not None:
                    # norm loss
                    # reg_offset = offset.norm(dim=-1, keepdim=True)
                    reg_offset = offset.abs()
                    if len(self.cfg.w.reg_offset_relax_for) > 0:
                        w = self.scale_vertex_weights_by_region(
                            weights=torch.ones_like(verts[:, :, :1]),
                            scale_factor=self.cfg.w.reg_offset_relax_coef,
                            region=self.cfg.w.reg_offset_relax_for,
                        )
                        reg_offset = w * reg_offset
                    log_dict["reg_offset"] = self.cfg.w.reg_offset * reg_offset.mean()
                
                if self.cfg.w.reg_offset_rigid is not None:
                    reg_offset_rigid = 0
                    for region in self.cfg.w.reg_offset_rigid_for:
                        vids = self.head.mask.get_vid_by_region([region])
                        reg_offset_rigid += offset[:, vids, :].var(dim=-2).mean()
                    log_dict["reg_offset_rigid"] = self.cfg.w.reg_offset_rigid * reg_offset_rigid

                if self.cfg.w.reg_offset_dynamic is not None and self.dynamic_offset is not None and self.opt_dict['dynamic_offset']:
                    # The dynamic offset is regularized to be temporally smooth
                    timesteps_prev = np.clip(timesteps - 1, 0, self.n_timesteps - 1)
                    reg_offset_d = self.dynamic_offset[timesteps_prev]
                    offset_d = self.dynamic_offset[timesteps]

                    reg_offset_dynamic = ((offset_d - reg_offset_d) ** 2).mean()
                    log_dict["reg_offset_dynamic"] = self.cfg.w.reg_offset_dynamic * reg_offset_dynamic

        return log_dict

    def scale_vertex_weights_by_region(self, weights, scale_factor, region):
        indices = self.head.mask.get_vid_by_region(region)
        weights[:, indices] *= scale_factor

        for _ in range(self.cfg.w.blur_iter):
            # multiply by the laplacian matrix with its diagonal negated
            diag = self.head.laplacian_diag[None, :, None]
            weights = (self.head.laplacian_smooth(weights) - 2 * diag * weights) / 2
        return weights

    def compute_pose_smooth_energy(self, timesteps):
        """
        Regularizes the global pose of the GNM head model to be temporally smooth
        """
        idx = timesteps
        idx_prev = np.clip(idx - 1, 0, self.n_timesteps - 1)

        E_trans = ((self.translation[idx] - self.translation[idx_prev].detach()) ** 2).mean() * self.cfg.w.smooth_trans
        E_rot = ((self.rotation[idx] - self.rotation[idx_prev].detach()) ** 2).mean() * self.cfg.w.smooth_rot
        return E_trans + E_rot
    
    def compute_joint_smooth_energy(self, timestep):
        """
        Regularizes the joints of the GNM head model to be temporally smooth
        """
        idx = timestep
        idx_prev = np.clip(idx - 1, 0, self.n_timesteps - 1)

        E_joint_smooth = 0
        E_joint_smooth += ((self.head_pose[idx] - self.head_pose[idx_prev].detach()) ** 2).mean() * self.cfg.w.smooth_head
        E_joint_smooth += ((self.eyes_pose[idx] - self.eyes_pose[idx_prev].detach()) ** 2).mean() * self.cfg.w.smooth_eyes
        return E_joint_smooth

    def compute_expr_smooth_energy(self, timestep):
        """
        Regularizes the expression of the GNM head model to be temporally smooth
        """
        idx = timestep
        idx_prev = np.clip(idx - 1, 0, self.n_timesteps - 1)

        E_expr_smooth = ((self.expr[idx] - self.expr[idx_prev].detach()) ** 2).mean() * self.cfg.w.smooth_expr
        return E_expr_smooth
    
    def compute_joint_L2_energy(self, timesteps):
        """
        Regularizes the joints of the GNM head model towards neutral joint locations
        """
        poses = [
            ("head", self.head_pose[timesteps, :]),
            ("eyes", self.eyes_pose[timesteps, :3]),
            ("eyes", self.eyes_pose[timesteps, 3:]),
        ]

        # Joints should are regularized towards neural
        E_joint_prior = 0
        for name, pose in poses:
            # L2 regularization for each joint
            rotmats = axis_angle_to_matrix(torch.cat([torch.zeros_like(pose), pose], dim=0))
            diff = ((rotmats[[0]] - rotmats[1:]) ** 2).mean()

            # Additional regularization for physical plausibility
            if name == 'eyes':
                # penalize the difference between the two eyes
                diff += ((self.eyes_pose[timesteps, :3] - self.eyes_pose[timesteps, 3:]) ** 2).mean()

            E_joint_prior += diff * self.cfg.w[f"reg_{name}"]
        return E_joint_prior

    def compute_laplacian_smoothing_loss(self, verts, offset_verts):
        basis_lap = self.head.laplacian_smooth(verts).detach()  #.norm(dim=-1) * weights

        offset_lap = self.head.laplacian_smooth(offset_verts)  #.norm(dim=-1) # * weights
        diff = (offset_lap - basis_lap) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff

    def compute_energy(
        self,
        sample,
        step_i=None,
        stage=None,
    ):
        """
        Compute total energy
        :param sample:
        frame energy
        :return: loss, log dict, predicted vertices and landmarks
        """
        log_dict = {}

        gt_rgb = sample["rgb"]
        result_dict = {"gt_rgb": gt_rgb}
        timesteps = sample["timestep_index"]

        verts, verts_cano, lmks, albedos = self.forward_head(timesteps)
        faces = self.head.faces

        if self.cfg.w.landmark is not None:
            if not self.cfg.w.always_enable_jawline_landmarks and stage is not None:
                disable_jawline_landmarks = self.cfg.pipeline[stage]['disable_jawline_landmarks']
            else:
                disable_jawline_landmarks = False
            E_lmk, _result_dict = self.compute_lmk_energy(sample, lmks, disable_jawline_landmarks)
            log_dict["lmk"] = self.cfg.w.landmark * E_lmk
            result_dict.update(_result_dict)
        
        if stage is None or isinstance(self.cfg.pipeline[stage], PhotometricStageConfig):
            if self.cfg.w.photo is not None:
                rast_dict = self.rasterize_head(
                    sample, verts, self.head.faces, train_mode=True
                )

                photo_energy_func = self.compute_photometric_energy
                E_photo, _result_dict = photo_energy_func(
                    sample,
                    verts,
                    faces, 
                    albedos,
                    rast_dict,
                    step_i,
                    stage,
                )
                result_dict.update(_result_dict)
                log_dict["photo"] = self.cfg.w.photo * E_photo

            enable_normal = stage is None or self.cfg.pipeline[stage].enable_normal
            if self.cfg.w.normal is not None and "normal" in sample and enable_normal:
                E_normal, _result_dict = self.compute_normal_energy(sample, verts, faces, rast_dict)
                result_dict.update(_result_dict)
                log_dict["normal"] = self.cfg.w.normal * E_normal

        if stage is not None:
            _log_dict = self.compute_regularization_energy(
                result_dict, verts, verts_cano, lmks, albedos, timesteps, stage
            )
            log_dict.update(_log_dict)

        E_total = torch.stack([v for k, v in log_dict.items()]).sum()
        log_dict["total"] = E_total

        return E_total, log_dict, verts, faces, lmks, albedos, result_dict

    @staticmethod
    def to_batch(x, indices):
        return torch.stack([x[i] for i in indices])

    @staticmethod
    def repeat_n_times(x: torch.Tensor, n: int):
        """Expand a tensor from shape [F, ...] to [F*n, ...]"""
        return x.unsqueeze(1).repeat_interleave(n, dim=1).reshape(-1, *x.shape[1:])

    @torch.no_grad()
    def log_scalars(
        self, 
        log_dict, 
        timestep, 
        session: Literal["train", "eval"] = "train", 
        stage=None,
        frame_step=None, 
        # step_in_stage=None, 
    ):
        """
        Logs scalars in log_dict to tensorboard and self.logger
        :param log_dict:
        :param timestep:
        :param step_i:
        :return:
        """

        if not self.calibrated and stage is not None and 'cam' in self.cfg.pipeline[stage].optimizable_params:
            log_dict["focal_length"] = self.focal_length.squeeze(0)

        log_msg = ""

        if session == "train":
            global_step = self.global_step
        else:
            global_step = timestep

        for k, v in log_dict.items():
            if not k.startswith("decay"):
                log_msg += "{}: {:.4f}  ".format(k, v)
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(f"{session}/{k}", v, global_step)

        if session == "train":
            assert stage is not None
            if frame_step is not None:
                msg_prefix = f"[{session}-{stage}] timestep {timestep} step {frame_step}:  "
            else:
                msg_prefix = f"[{session}-{stage}] timestep {timestep} step {self.global_step}:  "
        elif session == "eval":
            msg_prefix = f"[{session}] timestep {timestep}:  "
        self.logger.info(msg_prefix + log_msg)

    def save_obj_with_texture(self, vertices, faces, uv_coordinates, uv_indices, albedos, obj_path, mtl_path, texture_path):
        # Save the texture image
        torchvision.utils.save_image(albedos, texture_path)

        # Create the MTL file
        with open(mtl_path, 'w') as f:
            f.write(get_mtl_content(texture_path.name))
        
        # Create the obj file
        with open(obj_path, 'w') as f:
            f.write(get_obj_content(vertices, faces, uv_coordinates, uv_indices, mtl_path.name))
    
    def async_func(func):
        """Decorator to run a function asynchronously"""
        def wrapper(*args, **kwargs):
            self = args[0]
            if self.cfg.async_func:
                thread = threading.Thread(target=func, args=args, kwargs=kwargs)
                thread.start()
            else:
                func(*args, **kwargs)
        return wrapper
    
    @torch.no_grad()
    @async_func
    def log_media(
        self,
        verts: torch.tensor,
        faces: torch.tensor,
        lmks: torch.tensor,
        albedos: torch.tensor,
        output_dict: dict,
        sample: dict,
        timestep: int,
        session: str,
        stage: Optional[str]=None,
        frame_step: int=None,
        epoch=None,
    ):
        """
        Logs current tracking visualization to tensorboard
        :param verts:
        :param lmks:
        :param sample:
        :param timestep:
        :param frame_step:
        :param show_lmks:
        :param show_overlay:
        :return:
        """
        tic = time.time()
        prepare_output_path = partial(
            self.prepare_output_path, 
            session=session, 
            timestep=timestep, 
            stage=stage, 
            step=frame_step,
            epoch=epoch,
        )

        """images"""
        if not self.cfg.w.always_enable_jawline_landmarks and stage is not None:
            disable_jawline_landmarks = self.cfg.pipeline[stage]['disable_jawline_landmarks']
        else:
            disable_jawline_landmarks = False
        img = self.visualize_tracking(verts, lmks, albedos, output_dict, sample, disable_jawline_landmarks=disable_jawline_landmarks)
        img_path = prepare_output_path(folder_name="image_grid", file_type=self.cfg.log.image_format)
        torchvision.utils.save_image(img, img_path)

        """meshes"""
        if os.environ.get("VHAP_NO_MESH_DUMP"):  # per-frame textured OBJs are ~10 MB each; skip on tight quotas
            return
        texture_path = prepare_output_path(folder_name="mesh", file_type=self.cfg.log.image_format)
        mtl_path = prepare_output_path(folder_name="mesh", file_type="mtl")
        obj_path = prepare_output_path(folder_name="mesh", file_type="obj")
    
        vertices = verts[0].detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()
        uv_coordinates = self.head.verts_uvs.cpu().numpy()
        uv_indices = self.head.textures_idx.cpu().numpy()
        self.save_obj_with_texture(vertices, faces, uv_coordinates, uv_indices, albedos[0], obj_path, mtl_path, texture_path)
        """"""
    
        # log_figure = self.visualize_head_multiview(verts, faces, albedos, sample)
        # self.save_image(
        #     log_figure,
        #     timestep,
        #     folder_name=f"{session}/head_multiview",
        #     step=frame_step,
        # )

        # log_figure = self.visualize_trajectory(sample)
        # self.tb_writer.add_image("translation_trajectory", log_figure, frame_stepd)
        # self.save_image(
        #     log_figure, timestep, folder_name=f"{session}/translation_trajectory", step=frame_step
        # )

        toc = time.time() - tic
        if stage is not None:
            msg_prefix = f"[{session}-{stage}] timestep {timestep}"
        else:
            msg_prefix = f"[{session}] timestep {timestep}"
        if frame_step is not None:
            msg_prefix += f" step {frame_step}"
        self.logger.info(f"{msg_prefix}:  Logging media took {toc:.2f}s")

    @torch.no_grad()
    def visualize_tracking(
        self,
        verts,
        lmks,
        albedos,
        output_dict,
        sample,
        return_imgs_seperately=False,
        disable_jawline_landmarks=False,
    ):
        """
        Visualizes the tracking result
        """
        if len(self.cfg.log.view_indices) > 0:
            view_indices = torch.tensor(self.cfg.log.view_indices)
        else:
            num_views = sample["rgb"].shape[0]
            if num_views > 1:
                step = (num_views - 1) // (self.cfg.log.max_num_views - 1)
                view_indices = torch.arange(0, num_views, step=step)
            else:
                view_indices = torch.tensor([0])
        num_views_log = len(view_indices)

        imgs = []

        # rgb
        gt_rgb = output_dict["gt_rgb"][view_indices].cpu()
        transfm = torchvision.transforms.Resize(gt_rgb.shape[-2:])
        imgs += [img[None] for img in gt_rgb]

        if "pred_rgb" in output_dict:
            pred_rgb = transfm(output_dict["pred_rgb"][view_indices].cpu())
            pred_rgb = torch.clip(pred_rgb, min=0, max=1)
            imgs += [img[None] for img in pred_rgb]

        if "error_rgb" in output_dict:
            error_rgb = transfm(output_dict["error_rgb"][view_indices].cpu())
            error_rgb = error_rgb.mean(dim=1) / 2 + 0.5
            cmap = matplotlib.colormaps["seismic"]  # cm.get_cmap was removed in matplotlib 3.9
            error_rgb = cmap(error_rgb.cpu())
            error_rgb = torch.from_numpy(error_rgb[..., :3]).to(gt_rgb).permute(0, 3, 1, 2)
            imgs += [img[None] for img in error_rgb]
        
        # cluster id
        if "cid" in output_dict:
            cid = transfm(output_dict["cid"][view_indices].cpu())
            cid = cid / cid.max()
            cid = cid.expand(-1, 3, -1, -1).clone()

            pred_alpha = transfm(output_dict["pred_alpha"][view_indices].cpu()).expand(-1, 3, -1, -1)
            bg = pred_alpha == 0
            cid[bg] = 1
            imgs += [img[None] for img in cid]
        
        # albedo
        if "albedo" in output_dict:
            albedo = transfm(output_dict["albedo"][view_indices].cpu())
            albedo = torch.clip(albedo, min=0, max=1)

            pred_alpha = transfm(output_dict["pred_alpha"][view_indices].cpu()).expand(-1, 3, -1, -1)
            bg = pred_alpha == 0
            albedo[bg] = 1
            imgs += [img[None] for img in albedo]
        
        # normal
        if "normal" in output_dict:
            normal = transfm(output_dict["normal"][view_indices].cpu())
            normal = torch.clip(normal/2+0.5, min=0, max=1)
            imgs += [img[None] for img in normal]
        
        # diffuse
        diffuse = None
        if self.cfg.render.lighting_type != 'constant' and "diffuse" in output_dict:
            diffuse = transfm(output_dict["diffuse"][view_indices].cpu())
            diffuse = torch.clip(diffuse, min=0, max=1)
            imgs += [img[None] for img in diffuse]
        
        # aa
        if "aa" in output_dict:
            aa = transfm(output_dict["aa"][view_indices].cpu())
            aa = torch.clip(aa, min=0, max=1)
            imgs += [img[None] for img in aa]

        # alpha
        if "gt_alpha" in output_dict:
            gt_alpha = transfm(output_dict["gt_alpha"][view_indices].cpu()).expand(-1, 3, -1, -1)
            imgs += [img[None] for img in gt_alpha]

        if "pred_alpha" in output_dict:
            pred_alpha = transfm(output_dict["pred_alpha"][view_indices].cpu()).expand(-1, 3, -1, -1)
            color_alpha = torch.tensor([0.2, 0.5, 1])[None, :, None, None]
            fg_mask = (pred_alpha > 0).float()
            if diffuse is not None:
                fg_mask *= diffuse
                w = 0.7
            overlay_alpha = fg_mask * (w * color_alpha * pred_alpha + (1-w) * gt_rgb) \
                + (1 - fg_mask) * gt_rgb
            imgs += [img[None] for img in overlay_alpha]

        if "error_alpha" in output_dict:
            error_alpha = transfm(output_dict["error_alpha"][view_indices].cpu())
            error_alpha = error_alpha.mean(dim=1) / 2 + 0.5
            cmap = matplotlib.colormaps["seismic"]  # cm.get_cmap was removed in matplotlib 3.9
            error_alpha = cmap(error_alpha.cpu())
            error_alpha = (
                torch.from_numpy(error_alpha[..., :3]).to(gt_rgb).permute(0, 3, 1, 2)
            )
            imgs += [img[None] for img in error_alpha]
        else:
            error_alpha = None
        
        # landmark
        vis_lmk = self.visualize_landmarks(gt_rgb, output_dict, view_indices, disable_jawline_landmarks)
        if vis_lmk is not None:
            imgs += [img[None] for img in vis_lmk]
        # ----------------
        num_types = len(imgs) // len(view_indices)
        
        if return_imgs_seperately:
            return imgs
        else:
            if self.cfg.log.stack_views_in_rows:
                imgs = [imgs[j * num_views_log + i] for i in range(num_views_log) for j in range(num_types)]
                imgs = torch.cat(imgs, dim=0).cpu()
                return torchvision.utils.make_grid(imgs, nrow=num_types)
            else:
                imgs = torch.cat(imgs, dim=0).cpu()
                return torchvision.utils.make_grid(imgs, nrow=num_views_log)
    
    @torch.no_grad()
    def visualize_landmarks(self, gt_rgb, output_dict, view_indices=torch.tensor([0]), disable_jawline_landmarks=False):
        h, w = gt_rgb.shape[-2:]
        unit = h / 750
        wh = torch.tensor([[[w, h]]])
        vis_lmk = None
        if "gt_lmk2d" in output_dict:
            gt_lmk2d = (output_dict['gt_lmk2d'][view_indices].cpu() * 0.5 + 0.5) * wh
            if disable_jawline_landmarks:
                gt_lmk2d = gt_lmk2d[:, 17:68]
            else:
                gt_lmk2d = gt_lmk2d[:, :68]
            vis_lmk = gt_rgb.clone() if vis_lmk is None else vis_lmk
            for i in range(len(view_indices)):
                vis_lmk[i] = plot_landmarks_2d(
                    vis_lmk[i].clone(),
                    gt_lmk2d[[i]], 
                    colors="green",
                    unit=unit,
                    input_float=True, 
                ).to(vis_lmk[i])
        if "pred_lmk2d" in output_dict:
            pred_lmk2d = (output_dict['pred_lmk2d'][view_indices].cpu() * 0.5 + 0.5) * wh
            if disable_jawline_landmarks:
                pred_lmk2d = pred_lmk2d[:, 17:68]
            else:
                pred_lmk2d = pred_lmk2d[:, :68]
            vis_lmk = gt_rgb.clone() if vis_lmk is None else vis_lmk
            for i in range(len(view_indices)):
                vis_lmk[i] = plot_landmarks_2d(
                    vis_lmk[i].clone(),
                    pred_lmk2d[[i]], 
                    colors="red",
                    unit=unit,
                    input_float=True, 
                ).to(vis_lmk[i])
        return vis_lmk
        
    @torch.no_grad()
    def evaluate(self, make_visualization=True, epoch=0):
        # always save parameters before evaluation
        self.save_result(epoch=epoch)

        self.logger.info("Started Evaluation")
        # vid_frames = []
        photo_loss = []
        for timestep in range(self.n_timesteps):

            sample = self.dataset.getitem_by_timestep(timestep)
            self.clear_cache()
            self.fill_cam_params_into_sample(sample)
            (
                E_total,
                log_dict,
                verts,
                faces,
                lmks,
                albedos,
                output_dict,
            ) = self.compute_energy(sample)

            self.log_scalars(log_dict, timestep, session="eval")
            photo_loss.append(log_dict["photo"].item())

            if make_visualization and timestep % int(os.environ.get("VHAP_EVAL_MEDIA_EVERY", "1")) == 0:
                self.log_media(
                    verts,
                    faces,
                    lmks,
                    albedos,
                    output_dict,
                    sample,
                    timestep,
                    session="eval",
                    epoch=epoch,
                )
        
        self.tb_writer.add_scalar(f"eval_mean/photo", np.mean(photo_loss), epoch)

    def prepare_output_path(self, session, timestep, folder_name, file_type, stage=None, step=None, epoch=None):
        if epoch is not None:
            output_folder = self.out_dir / f'{session}_{epoch}' / folder_name
        else:
            output_folder = self.out_dir / session / folder_name
        os.makedirs(output_folder, exist_ok=True)
        
        if stage is not None:
            assert step is not None
            fname = "frame_{:05d}_{:03d}_{}.{}".format(timestep, step, stage, file_type)
        else:
            fname = "frame_{:05d}.{}".format(timestep, file_type)
        return output_folder / fname

    # def save_video(self, vid_frames, name):
    #     """Save image frames as a video.

    #     Args:
    #         vid_frame:
    #             type: torch.Tensor
    #             shape: (num_frames, height, width, 3)
    #             dtype: torch.uint8
    #         name: str
    #     """
    #     vid_path = str(self.out_dir / f"{name}.mp4")
    #     torchvision.io.write_video(
    #         vid_path,
    #         vid_frames,
    #         fps=self.config["frame_rate"],
    #         options={"crf": "10"},
    #     )


    def save_result(self, fname=None, epoch=None):
        """
        Saves tracked/optimized GNM parameters.
        :return:
        """
        # save parameters
        keys = [
            "rotation",
            "translation",
            "head_pose",
            "eyes_pose",
            "identity",
            "expr",
            "timestep_id",
            "n_processed_frames",
        ]
        values = [
            self.rotation,
            self.translation,
            self.head_pose,
            self.eyes_pose,
            self.identity,
            self.expr,
            np.array(self.dataset.timestep_ids),
            self.timestep,
        ]
        if not self.calibrated:
            keys += ["focal_length"]
            values += [self.focal_length]

        keys += ["tex"]
        values += [self.tex]

        if self.lights is not None:
            keys += ["lights"]
            values += [self.lights]
        
        # save the offsets the mesh actually used (coupled, if enabled), so exporters/avatars need no coupling logic
        couple = self.head.couple_offsets if self.cfg.model.couple_inner_offsets else (lambda x: x)
        if self.cfg.model.use_static_offset:
            keys += ["static_offset"]
            values += [couple(self.static_offset)]

        if self.cfg.model.use_dynamic_offset:
            keys += ["dynamic_offset"]
            values += [couple(self.dynamic_offset)]

        export_dict = {}
        for k, v in zip(keys, values):
            if not isinstance(v, np.ndarray):
                if isinstance(v, list):
                    v = torch.stack(v)
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
            export_dict[k] = v

        export_dict["image_size"] = np.array(self.image_size)

        fname = fname if fname is not None else "tracked_gnm_params"
        if epoch is not None:
            fname = f"{fname}_{epoch}"
        np.savez(self.out_dir / f'{fname}.npz', **export_dict)


class GlobalTracker(GnmTracker):
    def __init__(self, cfg: BaseTrackingConfig):
        super().__init__(cfg)

        self.calibrated = cfg.data.calibrated

        self.detect_landmarks(cfg)
        if cfg.data.use_normal:
            self.detect_normals(cfg)

        # logging
        out_dir = cfg.exp.output_folder / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir.mkdir(parents=True)

        self.timestep = self.cfg.begin_timestep
        self.out_dir = out_dir
        self.tb_writer = SummaryWriter(self.out_dir)
        
        self.log_interval_scalar = self.cfg.log.interval_scalar
        self.log_interval_media = self.cfg.log.interval_media

        config_yaml_path = out_dir / 'config.yml'
        config_yaml_path.write_text(yaml.dump(cfg), "utf8")
        print(tyro.to_yaml(cfg))

        self.logger = get_logger(__name__, root=True, log_dir=out_dir)

        # data
        self.dataset = import_module(cfg.data._target)(
            cfg=cfg.data,
            img_to_tensor=True,
        )
        # GnmTracker expects all views of a frame in a batch, which is undertaken by the
        # dataset. Therefore batching is disabled for the dataloader

        self.image_size = self.dataset[0]["rgb"].shape[-2:]
        self.n_timesteps = len(self.dataset)

        # parameters
        self.init_params()

        if self.cfg.model.gnm_params_path is not None:
            self.load_from_tracked_gnm_params(self.cfg.model.gnm_params_path)

    def detect_landmarks(self, cfg):
        cfg_data = deepcopy(cfg.data)
        cfg_data.use_landmark = False
        cfg_data.use_normal = False  # normals are annotated after landmarks; don't load them here
        dataset = import_module(cfg.data._target)(cfg=cfg_data, batchify_all_views=False)

        if cfg.data.landmark_source == 'face-alignment':
            if not cfg.exp.reuse_landmarks or not dataset.get_property_path("landmark2d/face-alignment", -1).exists():
                from vhap_gnm.util.landmark_detector_fa import annotate_landmarks
                annotate_landmarks(dataset, n_jobs=cfg.data.landmark_detector_njobs)
        elif cfg.data.landmark_source == 'star':
            if not cfg.exp.reuse_landmarks or not dataset.get_property_path("landmark2d/STAR", -1).exists():
                from vhap_gnm.util.landmark_detector_star import annotate_landmarks
                annotate_landmarks(dataset, n_jobs=cfg.data.landmark_detector_njobs)
        else:
            raise ValueError(f"Unknown landmark source: {cfg.data.landmark_source}")

    def detect_normals(self, cfg):
        assert cfg.data.normal_checkpoint is not None, \
            "cfg.data.normal_checkpoint must be set when cfg.data.use_normal is True"
        cfg_data = deepcopy(cfg.data)
        cfg_data.use_normal = False
        dataset = import_module(cfg.data._target)(cfg=cfg_data, batchify_all_views=False)

        if not cfg.exp.reuse_landmarks or not dataset.get_property_path("normal", -1).exists():
            from vhap_gnm.util.normal_detector import annotate_normals
            annotate_normals(
                dataset, cfg.data.normal_checkpoint,
                arch=cfg.data.normal_arch, device=cfg.data.normal_device,
            )

    def init_params(self):
        train_tensors = []

        # GNM model params
        self.identity = torch.zeros(self.head.n_identity_params).to(self.device)
        self.expr = torch.zeros(self.n_timesteps, self.head.n_expr_params).to(self.device)

        # joint axis angles
        self.head_pose = torch.zeros(self.n_timesteps, 3).to(self.device)
        self.eyes_pose = torch.zeros(self.n_timesteps, 6).to(self.device)

        # rigid pose
        # initialize the translation so that the head center starts at the origin
        head_center = self.head.v_template.mean(dim=0)
        self.translation = (-head_center)[None, :].repeat(self.n_timesteps, 1).contiguous()
        self.rotation = torch.zeros(self.n_timesteps, 3).to(self.device)

        # texture and lighting params
        res = self.cfg.model.tex_resolution
        self.tex = torch.full((3, res, res), 0.5).to(self.device)
        
        if self.cfg.render.lighting_type == 'SH':
            self.lights_uniform = torch.zeros(9, 3).to(self.device)
            self.lights_uniform[0] = torch.tensor([np.sqrt(4 * np.pi)]).expand(3).float().to(self.device)
            self.lights = self.lights_uniform.clone()
        else:
            self.lights = None

        train_tensors += (
            [self.identity, self.translation, self.rotation, self.head_pose, self.eyes_pose, self.expr,]
        )

        train_tensors += [self.tex]

        if self.lights is not None:
            train_tensors += [self.lights]

        if self.cfg.model.use_static_offset:
            self.static_offset = torch.zeros(1, self.head.v_template.shape[0], 3).to(self.device)
            train_tensors += [self.static_offset]
        else:
            self.static_offset = None

        if self.cfg.model.use_dynamic_offset:
            self.dynamic_offset = torch.zeros(self.n_timesteps, self.head.v_template.shape[0], 3).to(self.device)
            train_tensors += self.dynamic_offset
        else:
            self.dynamic_offset = None

        # camera definition
        if not self.calibrated:
            # K contains focal length and principle point
            self.focal_length = torch.tensor([1.5]).to(self.device)
            self.RT = torch.eye(3, 4).to(self.device)
            self.RT[2, 3] = -1  # (0, 0, -1) in w2c corresponds to (0, 0, 1) in c2w
            train_tensors += [self.focal_length]

        for t in train_tensors:
            t.requires_grad = True

    _CKPT_TENSORS = ("identity", "expr", "rotation", "translation", "head_pose", "eyes_pose", "tex", "lights",
                     "static_offset", "focal_length")

    def save_checkpoint(self, phase, progress, optimizer=None, scheduler=None):
        """Raw tracked tensors + where to continue (phase 'sequential': next timestep; 'global': epochs done)."""
        path = self.cfg.exp.output_folder / "checkpoint.pt"
        ck = dict(phase=phase, progress=progress, global_step=self.global_step,
                  tensors={k: getattr(self, k).detach().cpu() for k in self._CKPT_TENSORS
                           if getattr(self, k, None) is not None})
        if optimizer is not None:
            ck["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            ck["scheduler"] = scheduler.state_dict()
        tmp = path.with_suffix(".tmp")
        torch.save(ck, tmp)
        os.replace(tmp, path)  # atomic: a kill mid-save leaves the previous checkpoint intact
        self.logger.info(f"Checkpoint: {phase} {progress}")

    def load_checkpoint(self):
        path = self.cfg.exp.output_folder / "checkpoint.pt"
        if not self.cfg.exp.checkpoint_every or not path.exists():
            return None
        ck = torch.load(path, map_location=self.device, weights_only=False)
        for k, v in ck["tensors"].items():
            getattr(self, k).data.copy_(v.to(self.device))
        self.global_step = ck["global_step"]
        self.logger.info(f"Resuming from checkpoint: {ck['phase']} {ck['progress']}")
        return ck

    def optimize(self):
        """
        Optimizes GNM parameters on all frames of the dataset with random rampling
        :return:
        """
        self.global_step = 0
        every = self.cfg.exp.checkpoint_every
        ck = self.load_checkpoint()

        if ck is None or ck["phase"] == "sequential":
            self.optimize_sequential(start=ck["progress"] if ck else 0)
            if every:
                self.save_checkpoint("global", 0)
            ck = None

        self.logger.info(f"Start global optimization of all frames")
        # global optimization with random sampling
        dataloader = DataLoader(
            self.dataset, 
            batch_size=self.cfg.batch_size if not self.dataset.batchify_all_views else None, 
            shuffle=True, 
            num_workers=4
        )
        if self.cfg.exp.photometric:
            self.optimize_stage(stage='rgb_global_tracking', dataloader=dataloader, lr_scale=0.1, resume=ck)
        else:
            self.optimize_stage(stage='lmk_global_tracking', dataloader=dataloader, lr_scale=0.1, resume=ck)

        self.logger.info("All done.")

    def optimize_sequential(self, start=0):
        every = self.cfg.exp.checkpoint_every
        # sequential optimization of timesteps
        self.logger.info(f"Start sequential tracking GNM in {self.n_timesteps} frames (from timestep {start})")
        dataset = self.dataset if start == 0 else torch.utils.data.Subset(self.dataset, range(start, len(self.dataset)))
        dataloader = DataLoader(
            dataset, 
            batch_size=self.cfg.batch_size if not self.dataset.batchify_all_views else None, 
            shuffle=False, 
            num_workers=4
        )
        for sample in dataloader:
            if sample["timestep_index"][0].item() == 0:
                self.optimize_stage('lmk_init_rigid', sample)
                self.optimize_stage('lmk_init_all', sample)
                if self.cfg.exp.photometric:
                    self.init_texture_from_view(sample)
                    self.optimize_stage('rgb_init_texture', sample)
                    self.optimize_stage('rgb_init_all', sample)
                    if self.cfg.model.use_static_offset:
                        self.optimize_stage('rgb_init_offset', sample)

            if self.cfg.exp.photometric:
                self.optimize_stage('rgb_sequential_tracking', sample)
            else:
                self.optimize_stage('lmk_sequential_tracking', sample)
            self.initialize_next_timtestep(sample["timestep_index"])
            t_next = sample["timestep_index"][-1].item() + 1
            if every and t_next % every == 0:
                self.save_checkpoint("sequential", t_next)
        
        self.evaluate(make_visualization=True, epoch=0)
    
    def optimize_stage(
            self, 
            stage: Literal['lmk_init_rigid', 'lmk_init_all', 'rgb_init_texture', 'rgb_init_all', 'rgb_init_offset', 'rgb_sequential_tracking', 'rgb_global_tracking'],
            sample = None,
            dataloader = None,
            lr_scale = 1.0,
            resume = None,
        ):
        params = self.get_train_parameters(stage)
        optimizer = self.configure_optimizer(params, lr_scale=lr_scale)

        if sample is not None:
            num_steps = self.cfg.pipeline[stage].num_steps
            for step_i in range(num_steps):
                self.optimize_iter(sample, optimizer, stage)
        else:
            assert dataloader is not None
            num_epochs = self.cfg.pipeline[stage].num_epochs
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            start_epoch = 0
            if resume is not None and resume["phase"] == "global":
                start_epoch = resume["progress"]
                if "optimizer" in resume:
                    optimizer.load_state_dict(resume["optimizer"])
                    scheduler.load_state_dict(resume["scheduler"])
            for epoch_i in range(start_epoch, num_epochs):
                self.logger.info(f"EPOCH {epoch_i+1} / {num_epochs}")
                for step_i, sample in enumerate(dataloader):
                    self.optimize_iter(sample, optimizer, stage)
                scheduler.step()
                if self.cfg.exp.checkpoint_every:
                    self.save_checkpoint("global", epoch_i + 1, optimizer, scheduler)

                if (epoch_i + 1) % 10 == 0:
                    self.evaluate(make_visualization=True, epoch=epoch_i+1)
    
    def optimize_iter(self, sample, optimizer, stage):
        # compute loss and update parameters
        self.clear_cache()

        self.fill_cam_params_into_sample(sample)
        (
            E_total,
            log_dict,
            verts,
            faces,
            lmks,
            albedos,
            output_dict,
        ) = self.compute_energy(sample, stage=stage,
        )
        optimizer.zero_grad()
        E_total.backward()
        optimizer.step()

        if self.cfg.w.expr_clip is not None:
            # Safety net: GNM's region-local expression blendshapes have no
            # jaw joint to physically bound mouth motion, and can extrapolate
            # into grotesque geometry if a single dimension runs away
            # (typically under weak-landmark or hard-expression frames).
            self.expr.data.clamp_(-self.cfg.w.expr_clip, self.cfg.w.expr_clip)
        if self.cfg.model.freeze_tongue_expr:
            if not hasattr(self, "_tongue_dims"):
                self._tongue_dims = self.head.tongue_expr_dims().to(self.expr.device)
                self.logger.info(f"Freezing tongue-only expression dims: {self._tongue_dims.tolist()}")
            self.expr.data[..., self._tongue_dims] = 0

        # log energy terms and visualize
        timestep = sample["timestep_index"][0]
        if (self.global_step+1) % self.log_interval_scalar == 0:
            self.log_scalars(
                log_dict, 
                timestep, 
                session="train", 
                stage=stage, 
                frame_step=self.global_step, 
            )

        if (self.global_step+1) % self.log_interval_media == 0:
            self.log_media(
                verts,
                faces,
                lmks,
                albedos,
                output_dict,
                sample,
                timestep, 
                session="train",
                stage=stage,
                frame_step=self.global_step,
            )
        del verts, faces, lmks, albedos, output_dict
        self.global_step += 1


    def get_train_parameters(
        self, stage: Literal['lmk_init_rigid', 'lmk_init_all', 'rgb_init_all', 'rgb_init_offset', 'rgb_sequential_tracking', 'rgb_global_tracking'],
    ):
        """
        Collects the parameters to be optimized for the current frame
        :return: dict of parameters
        """
        self.opt_dict = defaultdict(bool)  # dict to keep track of which parameters are optimized
        for p in self.cfg.pipeline[stage].optimizable_params:
            self.opt_dict[p] = True
        
        params = defaultdict(list)  # dict to collect parameters to be optimized
            
        # shared properties
        if self.opt_dict["cam"] and not self.calibrated:
            params["cam"] = [self.focal_length]

        if self.opt_dict["identity"]:
            params["identity"] = [self.identity]

        if self.opt_dict["texture"]:
            params["tex"] = [self.tex]

        if self.opt_dict["static_offset"] and self.cfg.model.use_static_offset:
            params["static_offset"] = [self.static_offset]
        
        if self.opt_dict["lights"] and self.lights is not None:
            params["lights"] = [self.lights]
            
        # per-frame properties
        if self.opt_dict["pose"]:
            params["translation"].append(self.translation)
            params["rotation"].append(self.rotation)

        if self.opt_dict["joints"]:
            params["eyes"].append(self.eyes_pose)
            params["head"].append(self.head_pose)

        if self.opt_dict["expr"]:
            params["expr"].append(self.expr)
        
        if self.opt_dict["dynamic_offset"] and self.cfg.model.use_dynamic_offset:
            params["dynamic_offset"].append(self.dynamic_offset)

        return params

    def initialize_next_timtestep(self, timesteps):
        timestep_stride = timesteps[-1].item() - timesteps[0].item() + 1

        t_src = timesteps[-1]
        for s in range(timestep_stride):
            t_tgt = t_src + s + 1
            if t_tgt < self.n_timesteps - 1:
                self.translation[t_tgt].data.copy_(self.translation[t_src])
                self.rotation[t_tgt].data.copy_(self.rotation[t_src])
                self.head_pose[t_tgt].data.copy_(self.head_pose[t_src])
                self.eyes_pose[t_tgt].data.copy_(self.eyes_pose[t_src])
                self.expr[t_tgt].data.copy_(self.expr[t_src])
                if self.cfg.model.use_dynamic_offset:
                    self.dynamic_offset[t_tgt].data.copy_(self.dynamic_offset[t_src])
