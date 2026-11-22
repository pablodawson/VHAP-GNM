# VHAP-GNM: Versatile Head Alignment with Adaptive Appearance Priors, based on GNM

A port of [VHAP](https://github.com/ShenhanQian/VHAP) from the FLAME head model to
[GNM](https://github.com/google/GNM) (Generative aNthropometric Model), Google's
parametric 3D statistical human head model.

## TL;DR

- A photometric optimization pipeline based on differentiable mesh rasterization, applied to human head alignment.
- A perturbation mechanism that implicitly extract and inject regional appearance priors adaptively during rendering, enabling alignment of regions purely based on their appearance consistency, such as the hair, ears, neck, and shoulders, where no pre-defined landmarks are available.
- Tracks GNM parameters (identity, expression, joint rotations, translation) in monocular or multi-view videos.

## Differences from VHAP (FLAME)

For an explanation of how the VHAP pipeline works and the reasoning behind each change below, see
**[How it works](doc/how_it_works.md)**.

| | VHAP (FLAME) | VHAP-GNM |
| --- | --- | --- |
| Identity | `shape` (300 PCA coeffs) | `identity` (253 coeffs, fixed) |
| Expression | `expr` (100 PCA coeffs) | `expr` (383 region-based coeffs, fixed) |
| Global pose | root joint rotation + translation | `neck` (root) joint rotation + translation |
| Neck/head joint | `neck_pose` | `head_pose` (GNM `head` joint) |
| Jaw | `jaw_pose` joint | none — jaw motion lives in the expression basis |
| Eyes | `eyes_pose` (6) | `eyes_pose` (6, GNM `left_eye`/`right_eye` joints) |
| Teeth | grafted proxy teeth | native GNM teeth, gums, tongue, and eyeball geometry |
| Texture | FLAME PCA / painted texture + residual | a single optimizable texture map, warm-started by baking frame 0's visible face into UV space before any photometric optimization runs |
| Vertex masks | `FLAME_masks.pkl` + custom masks | GNM vertex groups + derived masks (`scalp`, `boundary`, `bottomline`, `neck`, `lips`, ...) |
| Model assets | manual download from MPI | shipped with the `gnm` package (Apache 2.0) |

The GNM expression basis is grouped by face region (not sorted by explained
variance), so truncating it is not meaningful; the full 253/383 dimensions are
always used and the `n_shape`/`n_expr` options were removed.

Options added in this fork (all off by default unless noted):

| Option | What it does |
| --- | --- |
| `--model.couple-inner-offsets` | mouth-interior vertices (gums, teeth, tongue, mouth sock) inherit the offsets of the surrounding skin, so per-vertex offsets can't push the cheeks through them. **Recommended.** |
| `--model.freeze-tongue-expr` | holds the tongue-only expression dims at 0. A monocular video never sees them, and otherwise the tongue pokes through the lips. **Recommended.** |
| `--w.expr-clip` (default 3.0) | hard clamp on the expression coefficients after every step. GNM has no jaw joint to bound mouth motion |
| `--data.use_normal`, `--w.normal`, `--w.normal-regions` | dense normal supervision from Sapiens2 (see [Discussions](#discussions)) |
| `VHAP_NO_MESH_DUMP=1` (env var) | skips the per-frame textured OBJ files written during evaluation (about 10 MB each) |

Since GNM has no jaw joint, mouth articulation is optimized through the
expression coefficients. Tracked parameters are saved to
`tracked_gnm_params_<epoch>.npz` with keys `identity`, `expr`, `rotation`
(neck/root joint), `head_pose`, `eyes_pose`, `translation`, `tex`, `lights`,
and optional `static_offset` / `dynamic_offset`.

## License

The original VHAP is made available under [CC-BY-NC-SA-4.0](./LICENSE) and is derived from the [multi-view head tracker of GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars/tree/main/reference_tracker), which is subjected to the following statements:

> Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to this software and related documentation. Any commercial use, reproduction, disclosure or distribution of this software and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.

GNM is licensed under Apache 2.0.

`vhap_gnm/third_party/sapiens2/` vendors code from Meta's [Sapiens2](https://github.com/facebookresearch/sapiens2) under the
[Sapiens2 License](vhap_gnm/third_party/sapiens2/LICENSE.md). That licence has its own use restrictions, and it applies to
that code and to the Sapiens2 checkpoints. It is only used by the optional normal loss.

## Setup

```shell
conda create --name VHAP-GNM -y python=3.12  # the gnm package requires Python >= 3.11
conda activate VHAP-GNM

# Install CUDA and ninja for compilation
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja cmake  # use the right CUDA version
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"  # to avoid error "/usr/bin/ld: cannot find -lcudart"
conda env config vars set CUDA_HOME=$CONDA_PREFIX  # for compilation

# Install PyTorch (make sure that the CUDA version matches with "Step 1")
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# make sure torch.cuda.is_available() returns True

# Install GNM (editable install recommended: the landmark data files are not shipped in wheels)
git clone https://github.com/google/GNM.git
pip install -e "GNM/gnm/shape[pytorch]"

pip install -e .
```

> [!NOTE]
> - We use an adjusted version of [nvdiffrast](https://github.com/ShenhanQian/nvdiffrast/tree/backface-culling) for backface-culling. If you have other versions installed before, you can reinstall as follows:
>    ```shell
>    pip install nvdiffrast@git+https://github.com/ShenhanQian/nvdiffrast@backface-culling --force-reinstall
>    rm -r ~/.cache/torch_extensions/*/nvdiffrast*
>    ```
> - We use [STAR](https://github.com/ShenhanQian/STAR/) for landmark detection by default. Alterntively, [face-alignment](https://github.com/1adrianb/face-alignment) is faster but less accurate.

## Troubleshooting

**If your shell sets a global `CUDA_HOME`/`CUDA_PATH` pointing at a different conda env's CUDA toolkit** (e.g. exported from `~/.bashrc` for some other project), any JIT-compiled CUDA extension (nvdiffrast) will pick up the wrong `nvcc` and fail with a CUDA-version mismatch or `-std` errors. Before running `track.py` or anything that touches the renderer, point these at your active env instead:

```shell
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH"   # only if cuda_runtime_api.h isn't directly under $CONDA_PREFIX/include
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.X/site-packages/torch/lib:$LD_LIBRARY_PATH"  # so the JIT-built .so can find libc10.so etc.
unset NVCC_PREPEND_FLAGS  # drop any host-compiler override set for a different toolkit
```

`source env.sh` (after `conda activate`) does the same.

The `gnm_pytorch.GNM` unhashable-dataclass issue, the nvdiffrast/newer-PyTorch module-loading incompatibility, and STAR's `scipy.integrate.simps` renaming are all patched inside this repo (`vhap_gnm/model/gnm.py`, `vhap_gnm/util/render_nvdiffrast.py`, `vhap_gnm/util/landmark_detector_star.py`) — no action needed for those.

## Download

### Video Data

#### Multiview

To get access to [NeRSemble](https://tobias-kirschstein.github.io/nersemble/) dataset, please request via the [Google Form](https://forms.gle/rYRoGNh2ed51TDWX9).

> [!NOTE]
> The NeRSemble dataset has been updated to Version 2. Its folder structure and color correction algorithm differ from those in Version 1, so please be careful not to confuse the two.

#### Monocular

We use monocular video sequences following [INSTA](https://zielon.github.io/insta/). You can download raw videos from [LRZ](https://syncandshare.lrz.de/getlink/fiJE46wKrG6oTVZ16CUmMr/VHAP).

## Usage

### Monocular
[For Monocular Videos](doc/monocular.md)

### Multiview
[For NeRSemble Dataset](doc/nersemble.md)

[For NeRSemble Dataset V2](doc/nersemble_v2.md)

## Discussions

Photometric alignment is versatile but sometimes sensitive.

**Texture map regularization**: Our method relies on a total-variation regularization on the texture map. Its loss weight is by default `1e4` for a monocualr video and `1e5` for the NeRSemble dataset (16 views). For you own multi-view dataset with fewer views, you should lower the regularization by passing `--w.reg_tex_tv 1e4` or 3e4. Otherwise, you may encounter corrupted shapes and blurry textures.

**Color affinity:** If the color of a point on the foreground contour is too close to the background, the `static_offset` can go wild. You may try a different background color by `--data.background_color white` or `--data.background_color black`. You can also disable `static_offset` by `--model.no_use_static_offset`.

**Occlussion:** When the neck is occluded by collars, the photometric gradients may squeeze and stretch the neck into unnatural shapes. Usually, this problem can be relieved by disabling photometric alignment in certain regions, e.g., `--model.occluded neck boundary`. Any GNM vertex group name can be used (see `vhap_gnm/model/gnm.py`), plus the derived regions `scalp`, `boundary`, `bottomline`, `neck`, `lips`, `left_ear`, and `right_ear`.

**Implausible bulges (e.g. on the cheeks):** landmarks don't cover the cheeks, so with only landmark + photometric loss, `static_offset`/`identity` there are only constrained by photometric residual, which alone is prone to explaining appearance with implausible geometry under monocular depth ambiguity — especially since `static_offset` is fit once from a single frame and then frozen for the rest of sequential tracking, so an error baked in from one view can look wrong from every other angle. Optional dense normal supervision from [Sapiens2](https://github.com/facebookresearch/sapiens2) (Meta) adds an independent, per-pixel geometric constraint everywhere the estimator is confident:

```shell
# one-time: download a normal-estimation checkpoint (0.4B is enough; larger = slower but more accurate)
mkdir -p ~/sapiens2_host/normal
curl -L "https://huggingface.co/facebook/sapiens2-normal-0.4b/resolve/main/sapiens2_0.4b_normal.safetensors" \
  -o ~/sapiens2_host/normal/sapiens2_0.4b_normal.safetensors

python vhap_gnm/track.py --data.root_folder data --data.sequence <seq> \
  --data.use_normal --data.normal_checkpoint ~/sapiens2_host/normal/sapiens2_0.4b_normal.safetensors \
  --w.normal 1.0 --w.normal-regions hockey_mask --w.reg-offset-lap 1e7
```

The first run caches one predicted normal map per frame under `<sequence>/normal/Sapiens2/` (reused on subsequent runs, like STAR landmarks). This only needs `torch` + `safetensors` + `opencv`/`numpy` — no `mmcv`/`mmengine` — via the vendored standalone backbone in `vhap_gnm/third_party/sapiens2/` (Meta's own ["no dependencies" usage mode](https://github.com/facebookresearch/sapiens2)). The loss weight `--w.normal` defaults to off (`None`). The values above worked best on our monocular data. With a
weight of 5, or with the scalp included in `--w.normal-regions`, the mesh crumpled into folds and hair spikes. The
loss only applies in the stages that optimise offsets (`rgb_init_offset`, sequential and global tracking), and only on
pixels inside the matted foreground.

## Interactive Viewers

Our method relies on vertex masks defined on GNM vertex groups. You can play with `regions` in our GNM Editor to see how each mask look like.

```shell
python vhap_gnm/gnm_editor.py
```

We also provide a GNM viewer for you to interact with a tracked sequence.

```shell
python vhap_gnm/gnm_viewer.py \
--param_path output/monocular/obama_whiteBg_staticOffset/2024-09-09_15-49-02/tracked_gnm_params_30.npz
```

Optional, you can enable colored rendering by specifying a texture image with `--tex_path`.

For both viewers, you can switch to flat shading with `--no-shade-smooth`.

## Cite

Please kindly cite the original VHAP repository and the GNM model if you find this software useful.

```bibtex
@misc{qian2024vhap,
  title={VHAP: Versatile Head Alignment with Adaptive Appearance Priors},
  author={Qian, Shenhan},
  year={2024},
  month={sep},
  doi={10.5281/zenodo.14988309}
  url={https://github.com/ShenhanQian/VHAP}
}
```
