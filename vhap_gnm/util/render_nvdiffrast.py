# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


import sys
from typing import Tuple, Literal, Optional
# from pytorch3d.structures.meshes import Meshes
import torch.utils.cpp_extension

# nvdiffrast._get_plugin() calls torch.utils.cpp_extension.load() and then
# separately does importlib.import_module(plugin_name), relying on load()
# having inserted the build directory into sys.path. Newer torch versions
# (>=2.x) import the compiled extension straight from its file path instead
# and never touch sys.path/sys.modules, so that second import fails with
# "No module named 'nvdiffrast_plugin'". Patch load() to keep doing that.
_cpp_extension_load = torch.utils.cpp_extension.load
def _cpp_extension_load_and_register_on_path(name, *args, **kwargs):
    module = _cpp_extension_load(name, *args, **kwargs)
    build_dir = torch.utils.cpp_extension._get_build_directory(name, False)
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    sys.modules.setdefault(name, module)
    return module
torch.utils.cpp_extension.load = _cpp_extension_load_and_register_on_path

import nvdiffrast.torch as dr
import torch.nn.functional as F
import torch
import numpy as np
from vhap_gnm.util import vector_ops as V


def get_SH_shading(normals, sh_coefficients, sh_const):
    """
    :param normals: shape N, H, W, K, 3
    :param sh_coefficients: shape N, 9, 3
    :return:
    """

    N = normals

    # compute sh basis function values of shape [N, H, W, K, 9]
    sh = torch.stack(
        [
            N[..., 0] * 0.0 + 1.0,
            N[..., 0],
            N[..., 1],
            N[..., 2],
            N[..., 0] * N[..., 1],
            N[..., 0] * N[..., 2],
            N[..., 1] * N[..., 2],
            N[..., 0] ** 2 - N[..., 1] ** 2,
            3 * (N[..., 2] ** 2) - 1,
        ],
        dim=-1,
    )
    sh = sh * sh_const[None, None, None, :].to(sh.device)

    # shape [N, H, W, K, 9, 1]
    sh = sh[..., None]

    # shape [N, H, W, K, 9, 3]
    sh_coefficients = sh_coefficients[:, None, None, :, :]

    # shape after linear combination [N, H, W, K, 3]
    shading = torch.sum(sh_coefficients * sh, dim=3)
    return shading


class NVDiffRenderer(torch.nn.Module):
    def __init__(
            self,
            use_opengl: bool = False, 
            lighting_type: Literal['constant', 'front', 'front-range', 'SH'] = 'front',
            lighting_space: Literal['camera', 'world'] = 'world',
            disturb_rate_fg: Optional[float] = 0.5,
            disturb_rate_bg: Optional[float] = 0.5,
            fid2cid: Optional[torch.Tensor] = None,
            shade_smooth: bool = True,
        ):
        super().__init__()
        self.backend = 'nvdiffrast'
        self.lighting_type = lighting_type
        self.lighting_space = lighting_space
        self.disturb_rate_fg = disturb_rate_fg
        self.disturb_rate_bg = disturb_rate_bg
        self.shade_smooth = shade_smooth
        self.glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        self.fragment_cache = None

        if fid2cid is not None:
            fid2cid = F.pad(fid2cid, [1, 0], value=0)  # for nvdiffrast, fid==0 means background pixels
            self.register_buffer("fid2cid", fid2cid, persistent=False)

        # constant factor of first three bands of spherical harmonics
        pi = np.pi
        sh_const = torch.tensor(
            [
                1 / np.sqrt(4 * pi),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
            ],
            dtype=torch.float32,
        )
        self.register_buffer("sh_const", sh_const, persistent=False)

    def clear_cache(self):
        self.fragment_cache = None
    
    def mvp_from_camera_param(self, RT, K, image_size):
        # projection matrix
        proj = self.projection_from_intrinsics(K, image_size)

        # Modelview and modelview + projection matrices.
        if RT.shape[-2] == 3:
            mv = torch.nn.functional.pad(RT, [0, 0, 0, 1])
            mv[..., 3, 3] = 1
        elif RT.shape[-2] == 4:
            mv = RT
        if proj.shape[0] < mv.shape[0]:
            proj = proj.expand(mv.shape[0], -1, -1)
        mvp = torch.bmm(proj, mv)
        return mvp
    
    def projection_from_intrinsics(self, K: torch.Tensor, image_size: Tuple[int], near: float=0.1, far:float=10):
        """
        Transform points from camera space (x: right, y: up, z: out) to clip space (x: right, y: down, z: in)
        Args:
            K: Intrinsic matrix, (N, 3, 3)
                K = [[
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1],
                    ]
                ]
            image_size: (height, width)
        Output:
            proj = [[
                    [2*fx/w, 0.0,     (w - 2*cx)/w,             0.0                     ],
                    [0.0,    2*fy/h, (h - 2*cy)/h,             0.0                     ],
                    [0.0,    0.0,     -(far+near) / (far-near), -2*far*near / (far-near)],
                    [0.0,    0.0,     -1.0,                     0.0                     ]
                ]
            ]
        """

        B = K.shape[0]
        h, w = image_size

        if K.shape[-2:] == (3, 3):
            fx = K[..., 0, 0]
            fy = K[..., 1, 1]
            cx = K[..., 0, 2]
            cy = K[..., 1, 2]
        elif K.shape[-1] == 4:
            fx, fy, cx, cy = K[..., [0, 1, 2, 3]].split(1, dim=-1)
        else:
            raise ValueError(f"Expected K to be (N, 3, 3) or (N, 4) but got: {K.shape}")

        proj = torch.zeros([B, 4, 4], device=K.device)
        proj[:, 0, 0]  = fx * 2 / w 
        proj[:, 1, 1]  = fy * 2 / h
        proj[:, 0, 2]  = (w - 2 * cx) / w
        proj[:, 1, 2]  = (h - 2 * cy) / h
        proj[:, 2, 2]  = -(far+near) / (far-near)
        proj[:, 2, 3]  = -2*far*near / (far-near)
        proj[:, 3, 2]  = -1
        return proj
    
    def world_to_camera(self, vtx, RT):
        """Transform vertex positions from the world space to the camera space"""
        RT = torch.from_numpy(RT).cuda() if isinstance(RT, np.ndarray) else RT
        if RT.shape[-2] == 3:
            mv = torch.nn.functional.pad(RT, [0, 0, 0, 1])
            mv[..., 3, 3] = 1
        elif RT.shape[-2] == 4:
            mv = RT

        # (x,y,z) -> (x',y',z',w)
        assert vtx.shape[-1] in [3, 4]
        if vtx.shape[-1] == 3:
            posw = torch.cat([vtx, torch.ones([*vtx.shape[:2], 1]).cuda()], axis=-1)
        elif vtx.shape[-1] == 4:
            posw = vtx
        else:
            raise ValueError(f"Expected 3D or 4D points but got: {vtx.shape[-1]}")
        return torch.bmm(posw, RT.transpose(-1, -2))
    
    def camera_to_clip(self, vtx, K, image_size):
        """Transform vertex positions from the camera space to the clip space"""
        K = torch.from_numpy(K).cuda() if isinstance(K, np.ndarray) else K
        proj = self.projection_from_intrinsics(K, image_size)
        
        # (x,y,z) -> (x',y',z',w)
        assert vtx.shape[-1] in [3, 4]
        if vtx.shape[-1] == 3:
            posw = torch.cat([vtx, torch.ones([*vtx.shape[:2], 1]).cuda()], axis=-1)
        elif vtx.shape[-1] == 4:
            posw = vtx
        else:
            raise ValueError(f"Expected 3D or 4D points but got: {vtx.shape[-1]}")

        if proj.shape[0] < posw.shape[0]:
            proj = proj.expand(posw.shape[0], -1, -1)
        return torch.bmm(posw, proj.transpose(-1, -2))
    
    def world_to_clip(self, vtx, RT, K, image_size):
        """Transform vertex positions from the world space to the clip space"""
        mvp = self.mvp_from_camera_param(RT, K, image_size)

        mvp = torch.from_numpy(mvp).cuda() if isinstance(mvp, np.ndarray) else mvp
        # (x,y,z) -> (x',y',z',w)
        posw = torch.cat([vtx, torch.ones([*vtx.shape[:2], 1]).cuda()], axis=-1)
        return torch.bmm(posw, mvp.transpose(-1, -2))
    
    def world_to_ndc(self, vtx, RT, K, image_size, flip_y=False):
        """Transform vertex positions from the world space to the NDC space"""
        verts_clip = self.world_to_clip(vtx, RT, K, image_size)
        verts_ndc = verts_clip[:, :, :3] / verts_clip[:, :, 3:]
        if flip_y:
            verts_ndc[:, :, 1] *= -1
        return verts_ndc

    def rasterize(self, verts, faces, RT, K, image_size, use_cache=False, require_grad=False):
        """
        Rasterizes meshes using a standard rasterization approach
        :param meshes:
        :param cameras:
        :param image_size:
        :return: fragments:
                 screen_coords: N x H x W x 2  with x, y values following pytorch3ds NDC-coord system convention
                                top left = +1, +1 ; bottom_right = -1, -1
        """
        # v_normals = self.compute_v_normals(verts, faces)
        # vertices and faces
        verts_camera = self.world_to_camera(verts, RT)
        verts_clip = self.camera_to_clip(verts_camera, K, image_size)
        tri = faces.int()
        rast_out, rast_out_db = self.rasterize_fragments(verts_clip, tri, image_size, use_cache, require_grad)
        rast_dict = {
            "rast_out": rast_out,
            "rast_out_db": rast_out_db,
            "verts": verts,
            "verts_camera": verts_camera[..., :3],
            "verts_clip": verts_clip,
        }
        
        # if not require_grad:
        #     verts_ndc = verts_clip[:, :, :3] / verts_clip[:, :, 3:]
        #     screen_coords = self.compute_screen_coords(rast_out, verts_ndc, faces, image_size)
        #     rast_dict["screen_coords"] = screen_coords

        return rast_dict

    def rasterize_fragments(self, verts_clip, tri, image_size, use_cache, require_grad=False):
        """
        Either rasterizes meshes or returns cached result
        """

        if not use_cache or self.fragment_cache is None:
            if require_grad:
                rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)
            else:
                with torch.no_grad():
                    rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)
            self.fragment_cache = (rast_out, rast_out_db)

        return self.fragment_cache

    def compute_screen_coords(self, rast_out: torch.Tensor, verts:torch.Tensor, faces:torch.Tensor, image_size: Tuple[int]):
        """ Compute screen coords for visible pixels
        Args:
            verts: (N, V, 3), the verts should lie in the ndc space 
            faces: (F, 3)
        """
        N = verts.shape[0]
        F = faces.shape[0]
        meshes = Meshes(verts, faces[None, ...].expand(N, -1, -1))
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        face_verts = verts_packed[faces_packed]

        # NOTE: nvdiffrast shifts face index by +1, and use 0 to flag empty pixel
        pix2face = rast_out[..., -1:].long() - 1  # (N, H, W, 1)
        is_visible = pix2face > -1  # (N, H, W, 1)
        # NOTE: is_visible is computed before packing pix2face to ensure correctness
        pix2face_packed = pix2face + torch.arange(0, N)[:, None, None, None].to(pix2face) * F

        bary_coords = rast_out[..., :2]  # (N, H, W, 2)
        bary_coords = torch.cat([bary_coords, 1 - bary_coords.sum(dim=-1, keepdim=True)], dim =-1)  # (N, H, W, 3)

        visible_faces = pix2face_packed[is_visible]  # (sum(is_visible), 3, 3)
        visible_face_verts = face_verts[visible_faces]
        visible_bary_coords = bary_coords[is_visible[..., 0]]  # (sum(is_visible), 3, 1)
        # visible_bary_coords = torch.cat([visible_bary_coords, 1 - visible_bary_coords.sum(dim=-1, keepdim=True)], dim =-1)

        visible_surface_point = visible_face_verts * visible_bary_coords[..., None]
        visible_surface_point = visible_surface_point.sum(dim=1)

        screen_coords = torch.zeros(*pix2face_packed.shape[:3], 2, device=meshes.device)
        screen_coords[is_visible[..., 0]] = visible_surface_point[:, :2]  # now have gradient

        return screen_coords
    
    def compute_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[..., i0, :]
        v1 = verts[..., i1, :]
        v2 = verts[..., i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        N = verts.shape[0]
        v_normals.scatter_add_(1, i0[..., None].repeat(N, 1, 3), face_normals)
        v_normals.scatter_add_(1, i1[..., None].repeat(N, 1, 3), face_normals)
        v_normals.scatter_add_(1, i2[..., None].repeat(N, 1, 3), face_normals)

        v_normals = torch.where(V.dot(v_normals, v_normals) > 1e-20, v_normals, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
        v_normals = V.safe_normalize(v_normals)
        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_normals))
        return v_normals
    
    def compute_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[..., i0, :]
        v1 = verts[..., i1, :]
        v2 = verts[..., i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = V.safe_normalize(face_normals)
        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(face_normals))
        return face_normals
    
    def shade(self, normal, lighting_coeff=None):
        if self.lighting_type == 'constant':
            diffuse = torch.ones_like(normal[..., :3])
        elif self.lighting_type == 'front':
            # diffuse = torch.clamp(V.dot(normal, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')), 0.0, 1.0)
            diffuse = V.dot(normal, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
            mask_backface = diffuse < 0
            diffuse[mask_backface] = diffuse[mask_backface].abs()*0.3
        elif self.lighting_type == 'front-range':
            bias = 0.75
            diffuse = torch.clamp(V.dot(normal, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')) + bias, 0.0, 1.0)
        elif self.lighting_type == 'SH':
            diffuse = get_SH_shading(normal, lighting_coeff, self.sh_const)
        else:
            raise NotImplementedError(f"Unknown lighting type: {self.lighting_type}")
        return diffuse
    
    def detach_by_indices(self, x, indices):
        x = x.clone()
        x[:, indices] = x[:, indices].detach()
        return x
    
    @torch.no_grad()
    def bake_texture_from_view(
        self, verts, faces, verts_uv, faces_uv, RT, K, gt_rgb, tex_resolution, fill_value=None,
    ):
        """
        Bakes a texture map by rendering the mesh from the given camera (the
        same rasterize+interpolate call used every training step to compute
        texture coordinates) and scattering each visible pixel's ground-truth
        color into its corresponding texel. Used to warm-start an optimizable
        texture with a plausible appearance instead of a flat color, so
        photometric optimization can focus on refining geometry from the start
        instead of also learning the entire facial appearance from scratch.
        Occlusion is handled for free by the camera-space z-buffer.

        Args:
            verts: (1, V, 3) posed vertex positions, world space.
            faces: (F, 3) int, position-vertex indices per face.
            verts_uv: (Vt, 2) uv coordinates in [0, 1], already v-flipped to match
                the convention used for forward texture sampling (dr.texture).
            faces_uv: (F, 3) int, uv-vertex indices per face (aligned with `faces`).
            RT: (1, 3, 4) world-to-camera extrinsics.
            K: (1, 4) or (1, 3, 3) intrinsics, in the format this class's
                `rasterize` expects.
            gt_rgb: (1, 3, H, W) ground-truth image, values in [0, 1].
            tex_resolution: int, output texture size (square).
            fill_value: (3,) tensor for uncovered texels, or None to use the mean
                of the covered texels.

        Returns:
            tex: (3, tex_resolution, tex_resolution)
            coverage: (1, tex_resolution, tex_resolution) bool
        """
        faces_i = faces.int()
        faces_uv_i = faces_uv.int()
        image_size = gt_rgb.shape[-2:]

        rast_dict = self.rasterize(verts, faces_i, RT, K, image_size, use_cache=False, require_grad=False)
        rast_out = rast_dict["rast_out"]
        rast_out_db = rast_dict["rast_out_db"]
        fg_mask = rast_out[..., -1] > 0  # (1, H, W), true only for visible (z-tested) surface

        # Same call render_rgba uses to get per-pixel texture coordinates.
        texc, _ = dr.interpolate(
            verts_uv[None, ...], rast_out, faces_uv_i, rast_db=rast_out_db, diff_attrs='all'
        )  # (1, H, W, 2), raw (pre-flip) nvdiffrast row order

        # rast_out/texc use nvdiffrast's raw row order; render_rgba flips its
        # final output by 1 row-axis to match gt_rgb's standard top-down rows,
        # so flip gt_rgb the same way here to pair each pixel with the right color.
        gt_rgb_hw3 = gt_rgb.flip(-2).permute(0, 2, 3, 1)  # (1, H, W, 3)

        col = (texc[..., 0] * (tex_resolution - 1)).round().long().clamp(0, tex_resolution - 1)
        row = (texc[..., 1] * (tex_resolution - 1)).round().long().clamp(0, tex_resolution - 1)
        fg_mask = fg_mask[0]  # (H, W)
        texel_idx = (row * tex_resolution + col)[0][fg_mask]  # (P,)
        colors = gt_rgb_hw3[0][fg_mask]  # (P, 3)

        num_texels = tex_resolution * tex_resolution
        accum = torch.zeros(num_texels, 3, device=verts.device)
        count = torch.zeros(num_texels, device=verts.device)
        accum.index_add_(0, texel_idx, colors)
        count.index_add_(0, texel_idx, torch.ones_like(texel_idx, dtype=torch.float32))

        covered_flat = count > 0
        if fill_value is None:
            fill_value = accum[covered_flat].sum(dim=0) / count[covered_flat].sum() if covered_flat.any() \
                else torch.full((3,), 0.5, device=verts.device)

        tex_flat = torch.where(
            covered_flat[:, None], accum / count[:, None].clamp(min=1), fill_value[None, :]
        )
        tex = tex_flat.reshape(1, tex_resolution, tex_resolution, 3).permute(0, 3, 1, 2)
        mask = covered_flat.reshape(1, 1, tex_resolution, tex_resolution).float()

        # The source photo is much lower-resolution than the texture atlas, so
        # only a sparse set of texels get hit directly. Dilate those samples
        # outward with repeated 3x3 averaging to fill the gaps between them,
        # so the whole visible-face region ends up covered, not just a sparse
        # scatter of isolated texels.
        for _ in range(40):
            if mask.all():
                break
            weighted_sum = F.avg_pool2d(tex * mask, 3, stride=1, padding=1) * 9
            mask_sum = F.avg_pool2d(mask, 3, stride=1, padding=1) * 9
            filled = weighted_sum / mask_sum.clamp(min=1e-6)
            tex = torch.where(mask.bool(), tex, filled)
            mask = torch.maximum(mask, (mask_sum > 0).float())

        coverage = mask[0].bool()  # after dilation: how much ended up with real (non-fallback) content
        tex = torch.where(mask.bool(), tex, fill_value.reshape(1, 3, 1, 1))
        return tex[0].contiguous(), coverage

    def render_rgba(
        self, rast_dict, verts, faces, verts_uv, faces_uv, tex, lights, background_color=[1., 1., 1.],
        align_texture_except_fid=None, align_boundary_except_vid=None, enable_disturbance=False,
    ):
        """
        Renders head RGBA images (for photometric optimization)
        """

        rast_out = rast_dict["rast_out"]
        rast_out_db = rast_dict["rast_out_db"]
        verts = rast_dict["verts"]
        verts_camera = rast_dict["verts_camera"]
        verts_clip = rast_dict["verts_clip"]
        faces = faces.int()
        faces_uv = faces_uv.int()
        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1).bool()

        out_dict = {}

        # ---- vertex attributes ----
        if  self.lighting_space == 'world':
            v_normal = self.compute_v_normals(verts, faces)
        elif  self.lighting_space == 'camera':
            v_normal = self.compute_v_normals(verts_camera, faces)
        else:
            raise NotImplementedError(f"Unknown lighting space: {self.lighting_space}")

        v_attr = [v_normal]
     
        v_attr = torch.cat(v_attr, dim=-1)
        attr, _ = dr.interpolate(v_attr, rast_out, faces)
        normal = attr[..., :3]
        normal = V.safe_normalize(normal)

        # ---- uv-space attributes ----
        texc, texd = dr.interpolate(verts_uv[None, ...], rast_out, faces_uv, rast_db=rast_out_db, diff_attrs='all')
        if align_texture_except_fid is not None:  # TODO: rethink when shading with normal
            fid = rast_out[..., -1:].long()  # the face index is shifted by +1
            mask = torch.zeros(faces.shape[0]+1, dtype=torch.bool, device=fid.device)
            mask[align_texture_except_fid + 1] = True
            b, h, w = rast_out.shape[:3]
            rast_mask = torch.gather(mask.reshape(1, 1, 1, -1).expand(b, h, w, -1), 3, fid)
            texc = torch.where(rast_mask, texc.detach(), texc)

        tex = tex.permute(0, 2, 3, 1).contiguous()  # (N, T, T, 4)
        albedo = dr.texture(tex, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=None)
        
        # ---- shading ----
        diffuse = self.shade(normal, lights)
        diffuse_detach_normal = self.shade(normal.detach(), lights)

        rgb = albedo * diffuse
        alpha = fg_mask.float()
        rgba = torch.cat([rgb, alpha], dim=-1)

        # ---- background ----
        if isinstance(background_color, list):
            """Background as a constant color"""
            rgba_bg = torch.tensor(background_color + [0]).to(rgba).expand_as(rgba)  # RGBA
        elif isinstance(background_color, torch.Tensor):
            """Background as a image"""
            rgba_bg = background_color
            rgba_bg = torch.cat([rgba_bg, torch.zeros_like(rgba_bg[..., :1])], dim=-1)  # RGBA
        else:
            raise ValueError(f"Unknown background type: {type(background_color)}")
        rgba_bg = rgba_bg.flip(1)  # opengl camera has y-axis up, needs flipping
        
        rgba = torch.where(fg_mask, rgba, rgba_bg)
        rgba_orig = rgba

        if enable_disturbance:
            # ---- color disturbance ----
            B, H, W, _ = rgba.shape
            # compute random blending weights based on the disturbance rate
            if self.disturb_rate_fg is not None:
                w_fg = (torch.rand_like(rgba[..., :1]) < self.disturb_rate_fg).int()
            else:
                w_fg = torch.zeros_like(rgba[..., :1]).int()
            if self.disturb_rate_bg is not None:
                w_bg = (torch.rand_like(rgba[..., :1]) < self.disturb_rate_bg).int()
            else:
                w_bg = torch.zeros_like(rgba[..., :1]).int()
            
            # sample pixles from clusters
            fid = rast_out[..., -1:].long()  # the face index is shifted by +1
            num_clusters = self.fid2cid.max() + 1

            fid2cid = self.fid2cid[None, None, None, :].expand(B, H, W, -1)
            cid = torch.gather(fid2cid, -1, fid)
            out_dict['cid'] = cid.flip(1)

            rgba_ = torch.zeros_like(rgba)
            for i in range(num_clusters):
                c_rgba = rgba_bg if i == 0 else rgba
                w = w_bg if i == 0 else w_fg

                c_mask = cid == i
                c_pixels = c_rgba[c_mask.repeat_interleave(4, dim=-1)].reshape(-1, 4).detach()  # NOTE: detach to avoid gradient flow

                if i != 1:  # skip #1 indicate faces that are not in any cluster
                    if len(c_pixels) > 0:
                        c_idx = torch.randint(0, len(c_pixels), (B * H * W, ), device=c_pixels.device)
                        c_sample = c_pixels[c_idx].reshape(B, H, W, 4)
                        rgba_ += c_mask * (c_sample * w + c_rgba * (1 - w))
                else:
                    rgba_ += c_mask * c_rgba
            rgba = rgba_

        # ---- AA on both RGB and alpha channels ----
        if align_boundary_except_vid is not None:
            verts_clip = self.detach_by_indices(verts_clip, align_boundary_except_vid)
        rgba_aa = dr.antialias(rgba, rast_out, verts_clip, faces.int())
        aa = ((rgba - rgba_aa) != 0).any(dim=-1, keepdim=True).repeat_interleave(4, dim=-1)

        # rgba_aa = torch.where(aa, rgba_aa, rgba_orig)  # keep the original color if not antialiased (commented out due to worse tracking performance)
        
        # ---- AA only on RGB channels ----
        # rgb = rgba[..., :3].contiguous()
        # alpha = rgba[..., 3:]
        # rgb = dr.antialias(rgb, rast_out, verts_clip, faces.int())
        # rgba = torch.cat([rgb, alpha], dim=-1)
        
        out_dict.update({
            'albedo': albedo.flip(1),
            'normal': normal.flip(1),
            'diffuse': diffuse.flip(1),
            'diffuse_detach_normal': diffuse_detach_normal.flip(1),
            'rgba': rgba_aa.flip(1),
            'aa': aa[..., :3].float().flip(1),
        })
        return out_dict

    def render_normal_camera_space(self, rast_dict, verts, faces, RT):
        """
        Renders a per-pixel unit camera/view-space normal map, independent of
        `self.lighting_space` (used for shading elsewhere). Matches the
        row/channel convention of gt_rgb (top-down) and of Sapiens2's camera-space
        normal predictions, for use as an external geometric supervision signal.

        Returns:
            normal: (N, H, W, 3) unit vectors, zero outside the foreground
            fg_mask: (N, H, W, 1) bool
        """
        rast_out = rast_dict["rast_out"]
        verts_camera = self.world_to_camera(verts, RT)[..., :3]
        v_normal = self.compute_v_normals(verts_camera, faces.int())
        normal, _ = dr.interpolate(v_normal, rast_out, faces.int())
        normal = V.safe_normalize(normal)
        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1).bool()
        normal = torch.where(fg_mask, normal, torch.zeros_like(normal))
        return normal.flip(1), fg_mask.flip(1)

    def render_vertex_region_mask(self, rast_dict, faces, vertex_mask):
        """
        Rasterizes a static per-vertex boolean region mask (e.g. "is this vertex
        part of the face/hair regions we trust for normal supervision") into a
        per-pixel mask, reusing an already-computed rasterization.

        Args:
            vertex_mask: (V,) bool/float, 1 for vertices inside the region
        Returns:
            (N, H, W, 1) bool, True where the interpolated region coverage is
            (near-)complete, i.e. the pixel is not a mix of in/out-of-region
            geometry straddling a region boundary.
        """
        rast_out = rast_dict["rast_out"]
        attr = vertex_mask.float().reshape(1, -1, 1).expand(rast_out.shape[0], -1, -1)
        region, _ = dr.interpolate(attr.contiguous(), rast_out, faces.int())
        region_mask = region > 0.999
        return region_mask.flip(1)

    def render_rgba_vis(
        self, verts, faces, RT, K, image_size, background_color=[1., 1., 1.],
        v_color=None, verts_uv=None, faces_uv=None, tex=None, lights=None,
    ):
        """
        Renders meshes into RGBA images (for visualization)
        """

        verts_camera_ = self.world_to_camera(verts, RT)
        verts_camera = verts_camera_[..., :3]
        verts_clip = self.camera_to_clip(verts_camera_, K, image_size)
        tri = faces.int()
        rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)

        faces = faces.int()
        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1).bool()
        face_id = torch.clamp(rast_out[..., -1:].long() - 1, 0)  # (B, W, H, 1)
        W, H = face_id.shape[1:3]

        if self.shade_smooth:
            if  self.lighting_space == 'world':
                v_normal = self.compute_v_normals(verts, faces)
            elif  self.lighting_space == 'camera':
                v_normal = self.compute_v_normals(verts_camera, faces)
            else:
                raise NotImplementedError(f"Unknown lighting space: {self.lighting_space}")

            normal, _ = dr.interpolate(v_normal, rast_out, faces)
            normal = V.safe_normalize(normal)
        else:
            face_normals = self.compute_face_normals(verts_camera, faces)  # (B, F, 3)
            face_normals_ = face_normals[:, None, None, :, :].expand(-1, W, H, -1, -1)  # (B, 1, 1, F, 3)
            face_id_ = face_id[:, :, :, None].expand(-1, -1, -1, -1, 3)  # (B, W, H, 1, 1)
            normal = torch.gather(face_normals_, -2, face_id_).squeeze(-2) # (B, W, H, 3)

        if verts_uv is not None and faces_uv is not None and tex is not None:
            texc, texd = dr.interpolate(verts_uv[None, ...], rast_out, faces_uv, rast_db=rast_out_db, diff_attrs='all')
            tex = tex.permute(0, 2, 3, 1).contiguous()  # (N, T, T, 4)
            albedo = dr.texture(tex, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=None)
        elif v_color is not None:
            v_attr = [v_color]
            v_attr = torch.cat(v_attr, dim=-1)
            attr, _ = dr.interpolate(v_attr, rast_out, faces)
            albedo = attr[..., :3]
        else:
            albedo = torch.ones_like(normal)
        
        # ---- shading ----
        if lights is not None:
            self.lighting_type = 'SH'
        diffuse = self.shade(normal, lights)

        rgb = albedo * diffuse
        alpha = fg_mask.float()
        rgba = torch.cat([rgb, alpha], dim=-1)

        # ---- background ----
        if isinstance(background_color, list) or isinstance(background_color, tuple):
            """Background as a constant color"""
            rgba_bg = torch.tensor(list(background_color) + [0]).to(rgba).expand_as(rgba)  # RGBA
        elif isinstance(background_color, torch.Tensor):
            """Background as a image"""
            rgba_bg = background_color
            rgba_bg = torch.cat([rgba_bg, torch.zeros_like(rgba_bg[..., :1])], dim=-1)  # RGBA
        else:
            raise ValueError(f"Unknown background type: {type(background_color)}")
        rgba_bg = rgba_bg.flip(1)  # opengl camera has y-axis up, needs flipping
        
        normal = torch.where(fg_mask, normal, rgba_bg[..., :3])
        diffuse = torch.where(fg_mask, diffuse, rgba_bg[..., :3])
        rgba = torch.where(fg_mask, rgba, rgba_bg)

        # ---- AA on both RGB and alpha channels ----
        rgba_aa = dr.antialias(rgba, rast_out, verts_clip, faces.int())
        
        return {
            'albedo': albedo.flip(1),
            'normal': normal.flip(1),
            'diffuse': diffuse.flip(1),
            'rgba': rgba_aa.flip(1),
            'verts_clip': verts_clip,
        }
