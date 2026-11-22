# How VHAP works, and what VHAP-GNM changes

This note explains the tracking pipeline inherited from [VHAP](https://github.com/ShenhanQian/VHAP).
It then covers every change made to run it on Google's [GNM](https://github.com/google/GNM) head model
instead of FLAME. File references point at this repository.

## 1. What VHAP does

VHAP fits a parametric 3D head to a video. Its outputs are:

- **A head model fitted to every frame:** identity, expression, pose and camera parameters.
- **Per-vertex geometry corrections** for that person.
- **A texture map and lighting.**
- **An optional export as a NeRF/3DGS-style dataset**, made of images, masks, `transforms.json` and per-frame
  parameters. Avatar methods such as GaussianAvatars train on this export.

It is an analysis-by-synthesis optimiser, not a learned regressor. Each step does four things:

1. It renders the textured, lit head mesh with a differentiable rasteriser (nvdiffrast).
2. It compares the render to the video frame.
3. It compares the projected mesh landmarks to 2D landmarks detected in the frame.
4. It back-propagates into the parameters.

### 1.1 Inputs, made once per sequence

| Input | Produced by | Used for |
| --- | --- | --- |
| RGB frames | `preprocess_video.py` (ffmpeg) | photometric loss |
| Foreground alpha | RobustVideoMatting, or BackgroundMattingV2 when a background plate exists | background for the render, masks for the export |
| 68 iBUG 2D landmarks | STAR (default) or face-alignment; cached under `<seq>/landmark2d/` | landmark loss |
| *(GNM fork only)* Normal maps | Sapiens2, optional; cached under `<seq>/normal/Sapiens2/` | normal loss |

### 1.2 What is optimised

| Group | Meaning |
| --- | --- |
| `cam` | camera intrinsics (focal length) |
| `pose` | global rotation and translation, one per frame |
| `identity` | head shape, shared by all frames |
| `joints` | neck/head and eye rotations, per frame |
| `expr` | expression coefficients, per frame |
| `texture` | one UV albedo map, shared |
| `lights` | spherical-harmonics lighting, shared |
| `static_offset` | a free 3D offset per vertex, shared. Adds detail the shape space can't represent: hair volume, nose, ears |
| `dynamic_offset` | a per-frame per-vertex offset; off by default |

### 1.3 Losses

- **Landmark:** distance between the projected mesh landmarks and the STAR landmarks.
  - Most photometric stages turn the jawline landmarks off, because 2D jawline detections slide along the contour.
  - `--w.always-enable-jawline-landmarks` (on by default) overrides that.
- **Photometric:** L1 between the rendered RGB and the frame. The render's background is the frame itself, so only
  pixels covered by the mesh produce colour error. The silhouette still gets gradient through anti-aliasing at the
  mesh edge.
- **Regularisers:**
  - L2 on identity, expression and joints;
  - total variation on the texture (`reg_tex_tv`), which keeps texture regions separate instead of smeared;
  - on `static_offset`: magnitude, Laplacian smoothness and local rigidity for ears, eyes and lips.
    Some of these are relaxed on the scalp and ears.
- **Temporal smoothness** on translation, rotation, joints and expression between neighbouring frames.

### 1.4 The schedule

Frame 0 initialises everything. Then each frame is tracked starting from the previous one. Finally, every frame is
refined jointly. The stages are in `vhap_gnm/config/base.py` and run in this order (`GnmTracker.optimize`):

| # | Stage | Frames | Optimises | Losses |
| --- | --- | --- | --- | --- |
| 1 | `lmk_init_rigid` | 0 | cam, pose | landmarks |
| 2 | `lmk_init_all` | 0 | + identity, joints, expr | landmarks |
| — | *texture bake (GNM fork)* | 0 | — | — |
| 3 | `rgb_init_texture` | 0 | cam, identity, texture, lights | + photometric |
| 4 | `rgb_init_all` | 0 | + pose, joints, expr | + photometric |
| 5 | `rgb_init_offset` | 0 | + static_offset | + photometric (+ normal) |
| 6 | `rgb_sequential_tracking` | each frame in order, starting from the previous frame's result | pose, joints, expr, texture | + photometric (+ normal) |
| 7 | `rgb_global_tracking` | random frames, 30 epochs, lr × 0.1 with exponential decay | everything | + photometric (+ normal) |

`--exp.no_photometric` swaps stages 3–7 for landmark-only variants. That is fast but coarse.

### 1.5 The key idea: adaptive appearance priors

Landmarks only cover the inner face. VHAP still aligns the hair, ears, neck and head outline, and it does so
photometrically with a trick in the renderer (`NVDiffRenderer.render_rgba`, `enable_disturbance`):

1. **Clusters.** Mesh faces are grouped into regions that should be roughly one colour (`--model.tex_clusters`:
   skin, scalp, boundary, lips, teeth, scleras, irises). The background is its own cluster.
2. **Perturbation.** During training, about 50 % of the rendered pixels in each cluster (`disturb_rate_fg`) are
   swapped for the colour of a random pixel from the same cluster. The background gets the same treatment
   (`disturb_rate_bg`). The swapped colours carry no gradient.
3. **Effect.** The texture can no longer explain a misalignment by painting fine detail. What still lowers the loss
   is moving region boundaries onto the image: hair edge against background, lip edge against skin, ear against
   hair. In effect each region's colour statistics act as a prior, estimated from the current texture and updated
   every step.

Two per-stage masks control this:

- **`align_texture_except`:** in these regions the texture lookup is detached from the geometry. The geometry then
  answers only to the silhouette, not to its interior colours. By default this covers the scalp, boundary and
  neck, because hair and clothing don't move with the mesh surface.
- **`align_boundary_except`:** these regions are excluded from edge (anti-aliasing) gradients. By default this is
  `bottomline`, the open bottom edge of the neck, so it isn't stretched to the bottom of the image.

`--model.occluded <regions>` adds regions to both lists, for example a neck hidden by a collar.

### 1.6 Export

`export_as_nerf_dataset.py` writes, per frame:

- the image and the alpha mask. `maskBelowLine` blanks everything below the neck cut, because the head model
  doesn't cover the body;
- the camera, in `transforms.json`;
- the fitted parameters, as `tracked_*_params` and `canonical.obj`.

GaussianAvatars-style methods attach Gaussians to the triangles of the tracked mesh.

## 2. From FLAME to GNM

### 2.1 Model mapping

| | VHAP (FLAME) | VHAP-GNM |
| --- | --- | --- |
| Shape | `shape`, 300 global PCA components | `identity`, 253 components |
| Expression | `expr`, 100 PCA components sorted by variance | `expr`, 383 **region-local** blendshapes: 0–99 left eye/brow, 100–199 right eye/brow, 200–349 mouth/jaw/nose/ears, 350–381 tongue, 382 iris. Not sorted by variance, so it is never truncated; `n_shape`/`n_expr` were removed |
| Jaw | `jaw_pose` joint | **none**: mouth opening lives in the expression basis |
| Global pose | root rotation + translation | rotation of GNM's root `neck` joint + translation |
| Neck | `neck_pose` | `head_pose`, GNM's `head` joint |
| Eyes | `eyes_pose` | `eyes_pose`, the `left_eye`/`right_eye` joints |
| Mouth interior | grafted proxy teeth | GNM's own teeth, gums, tongue, mouth sock and eyeballs |
| Texture | FLAME PCA or painted texture + residual | one optimisable UV map, warm-started from frame 0 (§2.3) |
| Masks | `FLAME_masks.pkl` | GNM vertex groups plus derived regions (§2.2) |
| Model files | manual download from MPI | shipped with the `gnm` package (Apache 2.0) |
| Output | `tracked_flame_params_*.npz` | `tracked_gnm_params_*.npz` with `identity, expr, rotation, head_pose, eyes_pose, translation, tex, lights[, static_offset, dynamic_offset]` |

All of this lives in `vhap_gnm/model/gnm.py` (`GnmHead`, `GnmMask`). FLAME code and the pytorch3d dependency are
gone: axis-angle conversion uses `gnm_common`, and there is a small sparse uniform Laplacian of our own.

### 2.2 Masks

VHAP refers to FLAME's named vertex regions throughout its configs. GNM has its own vertex groups, so the regions
that are missing are rebuilt from mesh topology (`GnmMask.create_custom_mask`):

- **`bottomline`:** vertices on the open bottom edge of the mesh, i.e. edges used by only one face.
- **`boundary`:** `bottomline` grown by 3 edge rings, which approximates FLAME's boundary strip.
- **`neck`:** vertices whose largest skinning weight is the neck joint.
- **`scalp`:** replaces FLAME's "hair". It is exterior skin that isn't face, ears or neck, **restricted to above the
  ears**. An earlier version also took in the jaw angle under the ears. That relaxed the offset regularisers there
  and switched off texture alignment, so the jawline drifted.
- **`lips`, `left_ear`, `right_ear`:** taken from GNM groups.

`python vhap_gnm/gnm_editor.py` shows every region on the mesh.

### 2.3 Fixes needed to run at all

| Problem | Fix |
| --- | --- |
| `gnm_pytorch.GNM` is both a dataclass and an `nn.Module`, so it is unhashable and `.to()` fails | `GNM.__hash__ = object.__hash__` (`model/gnm.py`) |
| `gnm_common.joint_transforms_world` creates helper tensors on the CPU, so it crashes on the GPU | a device-safe reimplementation is patched in (`model/gnm.py`) |
| nvdiffrast's JIT plugin loader relies on behaviour that newer PyTorch removed | `cpp_extension.load` is wrapped to register the module (`util/render_nvdiffrast.py`) |
| STAR imports `scipy.integrate.simps`, which was removed from scipy | a shim (`util/landmark_detector_star.py`) |
| UV vertices were deduplicated by `(u, v)` alone, merging about 1k different 3D vertices that happen to share a UV | the deduplication key is now `(vertex id, u, v)` |
| The package used the same import name (`vhap`) as stock VHAP, so imports mixed randomly when both were installed | renamed to `vhap_gnm` |

### 2.4 Changes for stability and quality

**No jaw joint means expression drift.** A FLAME jaw rotation is bounded. GNM's mouth blendshapes are not, and can
extrapolate into impossible shapes. Each frame also starts from the previous frame's values, so the error compounds.
Two fixes:

- `reg_expr` raised from 0.03 to 0.15;
- a hard clamp after every step, `--w.expr_clip`, default ±3.

**Texture warm start.** Before `rgb_init_texture`, frame 0's visible face is baked into the UV map:

1. The same camera-space rasterisation used for training gives the UV coordinate of each pixel.
2. Each foreground pixel's colour is scattered into its texel.
3. The gaps are dilated, and the rest is filled with the mean colour.

The photometric stages then start from a recognisable face rather than flat grey
(`GnmTracker.init_texture_from_view`).

**Sapiens2 normal loss.** This is optional; enable it with `--data.use_normal --w.normal …`.

- *Problem.* No landmarks cover the cheeks, so `static_offset` there is constrained only by colour. Under monocular
  depth ambiguity that can bake in bulges which show from other angles. Stock FLAME VHAP has the same bug.
- *Loss.* A dense loss, 1 − cos, between the rendered camera-space normals and Meta's Sapiens2 normal estimates.
  Sapiens2's axis convention was checked against our renderer, and no flip is needed.
- *When it applies.* Only in the stages that optimise offsets (5–7). Pixels must be inside both the rendered
  silhouette and the matted foreground, and inside `--w.normal_regions`.
- *Code.* The Sapiens2 backbone is vendored without its mmcv dependencies (`vhap_gnm/third_party/sapiens2`, under
  its own licence). The normal cache is saved as float16.

**Offsets leaking through the mouth interior** (`--model.couple-inner-offsets`). GNM's mouth interior (gums, teeth,
tongue, mouth sock) is a separate layer just behind the cheeks and lips. Free per-vertex offsets on the skin pushed
it through the interior, so gums showed through the cheek in renders. With the flag on, interior vertices inherit
the offsets of their 8 nearest exterior-skin vertices, weighted by inverse distance. The interior then moves with
the skin. Saved results contain the coupled offsets, so downstream code needs no change.

**Tongue poking out** (`--model.freeze-tongue-expr`). Expression dims 350–357 move only the tongue. A monocular
video never sees the tongue, so the tracker used those dims as spare freedom for the lower face. They saturated at
the clamp, and the tongue came out through the lips and chin on open-mouth frames. The flag holds them at zero.
The dims are detected automatically (`GnmHead.tongue_expr_dims`).

**Disk.** Setting `VHAP_NO_MESH_DUMP=1` skips the per-frame textured OBJ files written during evaluation, which are
about 10 MB each.

### 2.5 Recommended monocular settings

These are the settings used for our best monocular run:

```shell
python vhap_gnm/track.py --data.root_folder data/monocular --data.sequence <seq> \
  --exp.output_folder output/monocular/<seq>_whiteBg_staticOffset \
  --data.use_normal --data.normal_checkpoint <path>/sapiens2_0.4b_normal.safetensors \
  --w.normal 1.0 --w.normal-regions hockey_mask --w.reg-offset-lap 1e7 \
  --model.couple-inner-offsets --model.freeze-tongue-expr
```

What each normal-loss setting does:

- `--w.normal 1.0`, with normal supervision limited to the face (`hockey_mask`): a weight of 5 over the scalp too
  crumpled the mesh into folds and hair spikes.
- `--w.reg-offset-lap 1e7`: a stronger Laplacian prior that keeps the offsets smooth.
