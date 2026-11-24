# Wrapper around Google's GNM head model (https://github.com/google/GNM) that
# exposes the interface expected by the VHAP tracker: a differentiable head
# model with 2D/3D landmarks, vertex/face region masks, indexed UVs, and a
# laplacian matrix for offset regularization.


from gnm.shape import gnm_common, gnm_landmarks, gnm_pytorch, gnm_xnp
from vhap_gnm.util.log import get_logger
from matplotlib import cm

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

logger = get_logger(__name__)

# gnm_pytorch.GNM is a dataclass (eq=True, frozen=False) as well as an
# nn.Module; Python's dataclass machinery sets __hash__ = None in that case,
# but nn.Module.to()/named_children() need submodules to be hashable for
# cycle detection. Make it hashable by identity, like nn.Module itself.
if gnm_pytorch.GNM.__hash__ is None:
    gnm_pytorch.GNM.__hash__ = object.__hash__


def _joint_transforms_world_from_rotation_matrices_device_safe(joints, rotation_matrices, translation,
                                                             joint_parent_indices):
    """Drop-in replacement for gnm_common.joint_transforms_world_from_rotation_matrices.

    The original builds `parent_indices_except_root` and the homogeneous
    [0, 0, 0, 1] row via `xnp.asarray(...)` without a device kwarg, so on
    PyTorch they default to CPU and clash with CUDA joints/rotations. This
    mirrors the original logic but places both on `joints.device`.
    """
    if not isinstance(joints, torch.Tensor):
        return _gnm_from_rotation_matrices_orig(joints, rotation_matrices, translation, joint_parent_indices)

    num_joints = len(joint_parent_indices)
    parent_indices_except_root = torch.asarray(
        joint_parent_indices[1:], dtype=torch.int32, device=joints.device
    )
    joint_parents = gnm_common.take(joints, parent_indices_except_root, axis=-2, xnp=torch)

    root_translation = joints[..., 0, :] + translation
    non_root_translations = joints[..., 1:, :] - joint_parents
    local_translations = torch.concatenate(
        [root_translation[..., None, :], non_root_translations], axis=-2
    )

    local_transforms_3x4 = torch.concatenate(
        [rotation_matrices, local_translations[..., None]], axis=-1
    )

    bottom_row = gnm_common.zeros_with_batch_dims(
        joints, 2, (num_joints, 1, 4), dtype=joints.dtype
    )
    bottom_row = bottom_row + torch.asarray(
        [0.0, 0.0, 0.0, 1.0], dtype=joints.dtype, device=joints.device
    )

    local_transforms = torch.concatenate([local_transforms_3x4, bottom_row], axis=-2)

    world_transforms_list = [local_transforms[..., 0, :, :]]
    for joint_index in range(1, num_joints):
        parent_transform = world_transforms_list[joint_parent_indices[joint_index]]
        current_transform = parent_transform @ local_transforms[..., joint_index, :, :]
        world_transforms_list.append(current_transform)

    return torch.stack(world_transforms_list, axis=-3)


def _joint_transforms_world_device_safe(joints, rotations, translation, joint_parent_indices):
    """Drop-in replacement for gnm_common.joint_transforms_world (axis-angle rotations)."""
    if not isinstance(joints, torch.Tensor):
        return _gnm_joint_transforms_world_orig(joints, rotations, translation, joint_parent_indices)
    return _joint_transforms_world_from_rotation_matrices_device_safe(
        joints, gnm_common.axis_angle_to_rotation_matrix(rotations), translation, joint_parent_indices
    )


# Newer GNM (Nov 2026) skins through joint_transforms_world_from_rotation_matrices; older checkouts only have
# joint_transforms_world. Patch whichever exist.
_gnm_joint_transforms_world_orig = gnm_common.joint_transforms_world
gnm_common.joint_transforms_world = _joint_transforms_world_device_safe
if hasattr(gnm_common, "joint_transforms_world_from_rotation_matrices"):
    _gnm_from_rotation_matrices_orig = gnm_common.joint_transforms_world_from_rotation_matrices
    gnm_common.joint_transforms_world_from_rotation_matrices = _joint_transforms_world_from_rotation_matrices_device_safe


# GNM decorates its geometry methods with etils' @enp.check_and_normalize_arrays. Under older torch (2.4, our CUDA
# images) that normalisation silently detaches torch inputs, so identity/expression got no gradient: tracking left
# every expression at exactly 0. For torch tensors call the undecorated implementation (plain torch math) instead.
def _torch_passthrough(decorated):
    raw = decorated.__wrapped__

    def method(self, *args, **kwargs):
        if any(isinstance(a, torch.Tensor) for a in (*args, *kwargs.values())):
            return raw(self, *args, **kwargs)
        return decorated(self, *args, **kwargs)
    method.__wrapped__ = raw
    method.__doc__ = decorated.__doc__
    return method


for _name in ("vertex_positions_bind_pose", "joint_positions_bind_pose", "compute_pose_correctives",
              "joint_transforms_world", "get_posed_joint_transforms"):
    _m = getattr(gnm_xnp.GNM, _name, None)
    if _m is not None and hasattr(_m, "__wrapped__"):
        setattr(gnm_xnp.GNM, _name, _torch_passthrough(_m))

# GNM joint order: neck (root), head, left_eye, right_eye
JOINT_NECK = 0
JOINT_HEAD = 1
JOINT_LEFT_EYE = 2
JOINT_RIGHT_EYE = 3

VERTEX_GROUP_THRESHOLD = 1e-4  # same threshold GNM uses for vertex_group_mask


class GnmHead(nn.Module):
    """
    Given GNM parameters this class generates a differentiable GNM head mesh
    and 2D/3D facial landmarks.

    Parameters are:
        identity: (B, 253) identity coefficients
        expr: (B, 383) expression coefficients
        rotation: (B, 3) axis-angle of the root (neck) joint, i.e. the global rotation
        head_pose: (B, 3) axis-angle of the head joint (relative to the neck)
        eyes_pose: (B, 6) axis-angle of the left and right eye joints
        translation: (B, 3) global translation
    """

    def __init__(
        self,
        include_mask=True,
        include_lbs_color=False,
        face_clusters=[],
        couple_inner_offsets=False,
    ):
        super().__init__()

        logger.info("Initializing GNM head model...")

        self.gnm = gnm_pytorch.GNM.from_local(
            version=gnm_pytorch.GNMMajorVersion.V3,
            variant=gnm_pytorch.GNMVariant.HEAD,
        )

        self.n_identity_params = self.gnm.identity_dim
        self.n_expr_params = self.gnm.expression_dim

        self.register_buffer("v_template", self.gnm.template_vertex_positions.clone(), persistent=False)
        self.register_buffer("faces", self.gnm.triangles.long(), persistent=False)

        # Build indexed UVs (verts_uvs + textures_idx) from GNM's per-triangle UVs.
        # Corners are deduplicated by (position-vertex id, uv coordinate) rather
        # than uv coordinate alone: a handful of distinct 3D vertices (~1% of the
        # mesh, likely from symmetry) happen to share an identical uv coordinate,
        # and deduplicating by uv value alone would merge them into one uv-vertex,
        # making the uv-vertex -> position-vertex correspondence ambiguous (this
        # doesn't affect forward texture sampling, which never inverts that
        # correspondence, but breaks anything that bakes/unprojects a texture).
        faces_np = self.faces.cpu().numpy()  # (T, 3)
        triangle_uvs = self.gnm.triangle_uvs.cpu().numpy()  # (T, 3, 2)
        corner_keys = np.concatenate(
            [faces_np[..., None].astype(np.float64), triangle_uvs.astype(np.float64)], axis=-1
        ).reshape(-1, 3)  # (T*3, 3): (v_id, u, v)
        _, unique_indices, textures_idx = np.unique(
            corner_keys, axis=0, return_index=True, return_inverse=True
        )
        verts_uvs = triangle_uvs.reshape(-1, 2)[unique_indices]
        textures_idx = textures_idx.reshape(-1, 3)
        self.register_buffer("verts_uvs", torch.tensor(verts_uvs, dtype=torch.float32), persistent=False)
        self.register_buffer("textures_idx", torch.tensor(textures_idx, dtype=torch.long), persistent=False)

        # landmark embedding (68 iBUG landmarks as vertex indices + weights)
        lmk_config = gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
        self.register_buffer("lmk_vertex_ids", torch.tensor(lmk_config.indices, dtype=torch.long))
        self.register_buffer("lmk_weights", torch.tensor(lmk_config.weights, dtype=torch.float32))

        if include_mask:
            self.mask = GnmMask(
                self.gnm,
                faces=self.faces,
                faces_t=self.textures_idx,
                face_clusters=face_clusters,
            )

        # sparse uniform laplacian matrix (the GNM mesh is too large for a dense one):
        # L[i, j] = 1/deg(i) for each edge (i, j), and L[i, i] = -1
        laplacian_matrix = self.build_uniform_laplacian(self.faces, self.v_template.shape[0])
        self.register_buffer("laplacian_indices", laplacian_matrix.indices(), persistent=False)
        self.register_buffer("laplacian_values", laplacian_matrix.values(), persistent=False)
        indices, values = laplacian_matrix.indices(), laplacian_matrix.values()
        diag = torch.zeros(self.v_template.shape[0])
        on_diag = indices[0] == indices[1]
        diag[indices[0][on_diag]] = values[on_diag]
        self.register_buffer("laplacian_diag", diag, persistent=False)

        if include_lbs_color:
            self.add_lbs_color()

        # Layer coupling for per-vertex offsets. GNM's mouth interior (gums, teeth, tongue, mouth sock) is a separate
        # surface inside the cheeks/lips; with free per-vertex offsets the skin can be pulled through it (seen as gums
        # showing through the cheek). With coupling, inner vertices don't get their own offset: they inherit an
        # inverse-distance average of the offsets of their k nearest outer-skin vertices (template space), so the
        # layers move together. Idempotent, so already-coupled offsets pass through unchanged.
        self.couple_inner_offsets = couple_inner_offsets
        if include_mask:
            inner = self.mask.get_vid_by_region(["mouth_sock", "gums", "tongue", "teeth"])
            is_inner = torch.zeros(self.v_template.shape[0], dtype=torch.bool)
            is_inner[inner] = True
            outer = self.mask.get_vid_by_region(["skin_exterior"])
            outer = outer[~is_inner[outer]]
            d = torch.cdist(self.v_template[inner].float(), self.v_template[outer].float())  # (Ni, No)
            dist, nn = d.topk(8, dim=1, largest=False)
            w = 1.0 / dist.clamp_min(1e-4)
            self.register_buffer("inner_vids", inner, persistent=False)
            self.register_buffer("inner_nn", outer[nn], persistent=False)  # (Ni, 8) vertex ids
            self.register_buffer("inner_nn_w", w / w.sum(1, keepdim=True), persistent=False)

    @torch.no_grad()
    def tongue_expr_dims(self, min_mm=1.0, ratio=3.0):
        """Expression dims that (on the template) move the tongue >min_mm per unit and >ratio x more than any outer
        skin vertex. The tongue is rarely visible in monocular video, so a tracker can use these dims as free
        parameters, pushing the tongue through the chin; freeze them at 0 (model.freeze_tongue_expr)."""
        E = self.n_expr_params
        idt = torch.zeros(E, self.n_identity_params, device=self.v_template.device)
        B = self.gnm.vertex_positions_bind_pose(idt, torch.eye(E, device=idt.device)) \
            - self.gnm.vertex_positions_bind_pose(idt[:1], torch.zeros(1, E, device=idt.device))
        dt = B[:, self.mask.get_vid_by_region("tongue").to(B.device)].norm(dim=-1).amax(1) * 1000
        ds = B[:, self.mask.get_vid_by_region("skin_exterior").to(B.device)].norm(dim=-1).amax(1) * 1000
        return torch.where((dt > min_mm) & (dt > ratio * ds))[0]

    def couple_offsets(self, offset):
        """(B, V, 3) offsets -> same, with mouth-interior rows replaced by the weighted skin offsets above them."""
        if offset is None or not hasattr(self, "inner_vids"):
            return offset
        inherited = (offset[:, self.inner_nn] * self.inner_nn_w[None, ..., None]).sum(-2)  # (B, Ni, 3)
        return offset.index_copy(1, self.inner_vids, inherited)

    @staticmethod
    def build_uniform_laplacian(faces, num_verts):
        edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
        edges = edges.sort(dim=1)[0]
        edges = edges.unique(dim=0)
        edges = torch.cat([edges, edges.flip(1)], dim=0)  # both directions

        deg = torch.zeros(num_verts).index_add_(0, edges[:, 0], torch.ones(edges.shape[0]))
        off_diag_values = 1.0 / deg[edges[:, 0]]

        indices = torch.cat([edges.T, torch.arange(num_verts)[None].expand(2, -1)], dim=1)
        values = torch.cat([off_diag_values, -torch.ones(num_verts)], dim=0)
        return torch.sparse_coo_tensor(indices, values, (num_verts, num_verts)).coalesce()

    @property
    def laplacian_matrix(self):
        return torch.sparse_coo_tensor(
            self.laplacian_indices, self.laplacian_values,
            (self.v_template.shape[0], self.v_template.shape[0]),
        )

    def laplacian_smooth(self, x):
        """Multiplies (B, V, C) tensors by the sparse laplacian matrix."""
        B, V, C = x.shape
        y = torch.sparse.mm(self.laplacian_matrix, x.permute(1, 0, 2).reshape(V, B * C))
        return y.reshape(V, B, C).permute(1, 0, 2)

    def add_lbs_color(self):
        lbs_weights = self.gnm.skinning_weights.T  # (V, J)
        num_joints = lbs_weights.shape[1]
        color_indices = np.array(range(num_joints))
        cmap = cm.get_cmap("Set1")
        colors = torch.tensor(cmap(color_indices)[:, :3]).to(lbs_weights)  # (num_joints, 3)
        lbs_color = lbs_weights @ colors
        self.register_buffer("lbs_color", lbs_color.float(), persistent=False)

    def forward(
        self,
        identity,
        expr,
        rotation,
        head_pose,
        eyes_pose,
        translation,
        return_landmarks=True,
        return_verts_cano=False,
        static_offset=None,
        dynamic_offset=None,
    ):
        """
        Input:
            identity: N X number of identity parameters
            expr: N X number of expression parameters
            rotation, head_pose: N x 3 axis-angle
            eyes_pose: N x 6 axis-angle (left eye, right eye)
            translation: N x 3
        return:
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        rotations = torch.stack(
            [rotation, head_pose, eyes_pose[:, :3], eyes_pose[:, 3:]], dim=1
        )  # (N, 4, 3), in GNM joint order: neck, head, left_eye, right_eye

        v_bind = self.gnm.vertex_positions_bind_pose(identity, expr)
        joints_bind = self.gnm.joint_positions_bind_pose(identity)

        # Add personal offsets
        if self.couple_inner_offsets:
            static_offset = self.couple_offsets(static_offset)
            dynamic_offset = self.couple_offsets(dynamic_offset)
        v_cano = v_bind
        if static_offset is not None:
            v_cano = v_cano + static_offset
        if dynamic_offset is not None:
            v_cano = v_cano + dynamic_offset

        v_posed = v_cano + self.gnm.compute_pose_correctives(rotations)
        vertices = self.gnm.vertex_positions_world(v_posed, joints_bind, rotations, translation)

        ret_vals = [vertices]

        if return_verts_cano:
            ret_vals.append(v_cano)

        if return_landmarks:
            landmarks = (vertices[:, self.lmk_vertex_ids] * self.lmk_weights[None, ..., None]).sum(dim=-2)
            ret_vals.append(landmarks)

        if len(ret_vals) > 1:
            return ret_vals
        else:
            return ret_vals[0]


class BufferContainer(nn.Module):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        main_str = super().__repr__() + '\n'
        for name, buf in self.named_buffers():
            main_str += f'    {name:20}\t{buf.shape}\t{buf.dtype}\n'
        return main_str

    def __iter__(self):
        for name, buf in self.named_buffers():
            yield name, buf

    def keys(self):
        return [name for name, buf in self.named_buffers()]

    def items(self):
        return [(name, buf) for name, buf in self.named_buffers()]


class GnmMask(nn.Module):
    """Vertex/face region masks built from GNM's vertex groups.

    Available regions include GNM's vertex groups (skin, skin_exterior, left,
    right, hockey_mask, upper_lip, lower_lip, mouth_sock, tongue, gums, teeth,
    eyes, left_eye, right_eye, eye_sockets, scleras, irises, pupils, ears, ...)
    plus custom ones: lips, left_ear, right_ear, neck, boundary, bottomline,
    scalp.
    """

    def __init__(
            self,
            gnm,
            faces,
            faces_t,
            face_clusters=[],
        ):
        super().__init__()
        self.faces = faces
        self.faces_t = faces_t
        self.face_clusters = face_clusters
        self.num_verts = gnm.num_vertices
        self.num_faces = faces.shape[0]

        self.process_vertex_mask(gnm)
        self.create_custom_mask(gnm)

        self.construct_vid_table()
        self.process_face_mask(self.faces)
        self.process_face_clusters(self.face_clusters)
        self.process_vt_mask(self.faces, self.faces_t)

    def process_vertex_mask(self, gnm):
        """Convert GNM vertex groups into vertex index buffers"""
        logger.info("Processing vertex masks for GNM...")

        self.v = BufferContainer()
        vertex_groups = gnm.vertex_groups.cpu()
        for i, name in enumerate(gnm.vertex_group_names):
            vid = torch.where(vertex_groups[i] > VERTEX_GROUP_THRESHOLD)[0]
            self.v.register_buffer(name, vid)

    def create_custom_mask(self, gnm):
        """Add custom masks derived from the GNM vertex groups and topology"""
        v = {name: buf for name, buf in self.v}

        # lips
        self.v.register_buffer("lips", torch.cat([v["upper_lip"], v["lower_lip"]]).unique())

        # per-side ears
        left = set(v["left"].tolist())
        ears = v["ears"]
        is_left = torch.tensor([vid.item() in left for vid in ears], dtype=torch.bool)
        self.v.register_buffer("left_ear", ears[is_left])
        self.v.register_buffer("right_ear", ears[~is_left])

        # neck: vertices dominated by the neck (root) joint
        skinning_segmentation = gnm.skinning_weights.argmax(dim=0)
        self.v.register_buffer("neck", torch.where(skinning_segmentation == JOINT_NECK)[0])

        # bottomline: vertices on the open bottom edge of the mesh. Edges that
        # appear in only one triangle are open; we keep those on the skin
        # component (eyes/teeth interiors have their own open edges).
        edges = torch.cat([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]], dim=0)
        edges = edges.sort(dim=1)[0]
        edge_keys = edges[:, 0] * self.num_verts + edges[:, 1]
        uniques, counts = edge_keys.unique(return_counts=True)
        open_edge_keys = uniques[counts == 1]
        open_vids = torch.cat([open_edge_keys // self.num_verts, open_edge_keys % self.num_verts]).unique()
        skin = torch.zeros(self.num_verts, dtype=torch.bool)
        skin[v["skin"]] = True
        bottomline = open_vids[skin[open_vids]]
        self.v.register_buffer("bottomline", bottomline)

        # boundary: a band of vertices around the open bottom edge, analogous to
        # FLAME's boundary strip (grown by a few edge rings from the bottomline)
        in_boundary = torch.zeros(self.num_verts, dtype=torch.bool)
        in_boundary[bottomline] = True
        for _ in range(3):
            on_edge = in_boundary[edges[:, 0]] | in_boundary[edges[:, 1]]
            in_boundary[edges[on_edge].reshape(-1)] = True
        self.v.register_buffer("boundary", torch.where(in_boundary)[0])

        # anchor points on the bottomline ring (+x is the subject's left, +z is front)
        bottomline_verts = gnm.template_vertex_positions[bottomline].cpu()
        self.v.register_buffer("neck_left_point", bottomline[bottomline_verts[:, 0].argmax()][None])
        self.v.register_buffer("neck_right_point", bottomline[bottomline_verts[:, 0].argmin()][None])
        self.v.register_buffer("front_middle_bottom_point_boundary", bottomline[bottomline_verts[:, 2].argmax()][None])

        # scalp: exterior skin that is not part of the face, ears, or neck
        # (plays the role of FLAME's "hair" region)
        keep = torch.zeros(self.num_verts, dtype=torch.bool)
        keep[v["skin_exterior"]] = True
        for name in ["hockey_mask", "ears"]:
            keep[v[name]] = False
        keep[self.v.get_buffer("neck")] = False
        # "everything else" also covered the jaw angle below the ears (~half the old scalp), which then got relaxed
        # smoothing and no texture alignment; keep only skin above mid-ear height (+y is up)
        y = gnm.template_vertex_positions[:, 1].cpu()
        keep &= y > y[v["ears"]].mean()
        self.v.register_buffer("scalp", torch.where(keep)[0])

    def construct_vid_table(self):
        self.vid_to_region = defaultdict(list)  # vertex id -> region name
        for region_name, v_mask in self.v:
            for v_id in v_mask:
                self.vid_to_region[v_id.item()].append(region_name)

    def process_face_mask(self, faces):
        logger.info("Processing face masks for GNM...")

        face_masks = defaultdict(list)  # region name -> face id
        for f_id, f in enumerate(faces):
            counters = defaultdict(int)
            for v_id in f:
                for region_name in self.vid_to_region[v_id.item()]:
                    counters[region_name] += 1

            for region_name, count in counters.items():
                if count >= 3:  # create straight boundaries, with seams
                    face_masks[region_name].append(f_id)

        self.f = BufferContainer()
        for region_name, f_mask in face_masks.items():
            self.f.register_buffer(region_name, torch.tensor(f_mask, dtype=torch.long))

    def process_face_clusters(self, face_clusters):
        """ Construct a lookup table from face id to cluster id.

            cluster #0: background
            cluster #1: foreground
            cluster #2: faces in face_clusters[0]
            cluster #3: faces in face_clusters[1]
            ...
        """
        logger.info("Processing face clusters...")

        fid2cid = torch.ones(self.num_faces+1, dtype=torch.long)  # faces are always treated as foreground
        for cid, cluster in enumerate(face_clusters):
            try:
                fids = self.get_fid_by_region([cluster])
            except Exception as e:
                logger.warning(f"Ignoring unknown cluster {cluster}.")
                continue
            fid2cid[fids] = cid + 2  # reserve cluster #0 for the background and #1 for faces that do not belong to any cluster
        self.register_buffer("fid2cid", fid2cid)

    def process_vt_mask(self, faces, faces_t):
        logger.info("Processing vt masks for GNM...")

        vt_masks = defaultdict(list)  # region name -> vt id
        for f_id, (face, face_t) in enumerate(zip(faces, faces_t)):
            for v_id, vt_id in zip(face, face_t):
                for region_name in self.vid_to_region[v_id.item()]:
                    vt_masks[region_name].append(vt_id.item())

        self.vt = BufferContainer()
        for region_name, vt_mask in vt_masks.items():
            self.vt.register_buffer(region_name, torch.tensor(vt_mask, dtype=torch.long))

    def get_vid_by_region(self, regions, keep_order=False):
        """Get vertex indicies by regions"""
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            vid = torch.cat([self.v.get_buffer(k) for k in regions])
            if keep_order:
                return vid
            else:
                return vid.unique()
        else:
            return torch.tensor([], dtype=torch.long)

    def get_vid_except_region(self, regions):
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            indices = torch.cat([self.v.get_buffer(k) for k in regions]).unique()
        else:
            indices = torch.tensor([], dtype=torch.long)

        # get the vertex indicies that are not included by regions
        vert_idx = torch.arange(0, self.num_verts, device=indices.device)
        combined = torch.cat((indices, vert_idx))
        uniques, counts = combined.unique(return_counts=True)
        return uniques[counts == 1]

    def get_fid_by_region(self, regions):
        """Get face indicies by regions"""
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            return torch.cat([self.f.get_buffer(k) for k in regions]).unique()
        else:
            return torch.tensor([], dtype=torch.long)

    def get_fid_except_region(self, regions):
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            indices = torch.cat([self.f.get_buffer(k) for k in regions]).unique()
        else:
            indices = torch.tensor([], dtype=torch.long)

        # get the face indicies that are not included by regions
        face_idx = torch.arange(0, self.num_faces, device=indices.device)
        combined = torch.cat((indices, face_idx))
        uniques, counts = combined.unique(return_counts=True)
        return uniques[counts == 1]

    def get_fid_except_fids(self, fids):
        # get the face indicies that are not included
        face_idx = torch.arange(0, self.num_faces, device=fids.device)
        combined = torch.cat((fids, face_idx))
        uniques, counts = combined.unique(return_counts=True)
        return uniques[counts == 1]
