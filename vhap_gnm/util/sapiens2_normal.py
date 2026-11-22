# Minimal-dependency inference wrapper for Sapiens2's surface-normal model
# (https://github.com/facebookresearch/sapiens2), used to supply an
# independent geometric supervision signal for regions that photometric +
# landmark loss alone under-constrain (e.g. cheeks, which have no iBUG-68
# landmarks). Only depends on torch + safetensors + numpy + cv2, matching the
# "no dependencies" standalone usage documented in the sapiens2 repo -- no
# mmcv/sapiens.engine config system is required.
#
# Preprocessing/postprocessing here mirror sapiens2's own
# NormalResizePadImage + ImagePreprocessor + tools/vis/vis_normal.py exactly
# (letterbox resize to 1024x768 preserving aspect ratio, center-padded,
# ImageNet mean/std normalization in [0, 255] scale, BGR->RGB), so that the
# same checkpoints produce the same output as the reference implementation.

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from vhap_gnm.third_party.sapiens2.backbone import Sapiens2
from vhap_gnm.third_party.sapiens2.normal_head import NormalHead

IMAGE_SIZE = (1024, 768)  # (height, width), native resolution for all arches
PATCH_SIZE = 16
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)

# gnm/shape/data/versions/v3_0/gnm_head.npz-style arch table: (embed_dims, num_layers, num_heads)
# only embed_dims is needed here (drives decode_head.in_channels); the rest is
# baked into the vendored Sapiens2 backbone's own arch_zoo.
_EMBED_DIMS = {
    "sapiens2_0.1b": 768,
    "sapiens2_0.4b": 1024,
    "sapiens2_0.8b": 1280,
    "sapiens2_1b": 1536,
    "sapiens2_5b": 2432,
}


class Sapiens2NormalModel(torch.nn.Module):
    """backbone + decode_head, matching the attribute names (and therefore
    the checkpoint's `backbone.*` / `decode_head.*` state_dict keys) of
    sapiens2's own `NormalEstimator`."""

    def __init__(self, arch="sapiens2_0.4b"):
        super().__init__()
        self.backbone = Sapiens2(
            arch=arch,
            img_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            final_norm=True,
            use_tokenizer=False,
            with_cls_token=True,
            out_type="featmap",
        )
        self.decode_head = NormalHead(
            in_channels=_EMBED_DIMS[arch],
            upsample_channels=[768, 512, 256, 128],
            conv_out_channels=[64, 32, 16],
            conv_kernel_sizes=[3, 3, 3],
        )

    def forward(self, x):
        features = self.backbone(x)[0]
        return self.decode_head(features)


def load_sapiens2_normal(checkpoint_path, arch="sapiens2_0.4b", device="cuda", dtype=torch.float32):
    model = Sapiens2NormalModel(arch=arch)
    state_dict = load_file(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Sapiens2 normal checkpoint did not match the model exactly "
            f"(missing={missing}, unexpected={unexpected}). The vendored "
            f"backbone/head architecture may be out of sync with the checkpoint."
        )
    return model.to(device=device, dtype=dtype).eval()


def _resize_pad(image_bgr, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]):
    """Letterbox resize (preserving aspect ratio) + center pad to (height, width).
    Mirrors sapiens2's NormalResizePadImage exactly."""
    import cv2

    orig_h, orig_w = image_bgr.shape[:2]
    scale = min(width / orig_w, height / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = width - new_w, height - new_h
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    padded = np.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    padding_size = (pad_left, pad_right, pad_top, pad_bottom)
    return padded, padding_size


def preprocess(image_bgr, device="cuda", dtype=torch.float32):
    """image_bgr: (H, W, 3) uint8, as returned by cv2.imread.
    Returns (input tensor (1, 3, 1024, 768), padding_size, orig_shape)."""
    padded_bgr, padding_size = _resize_pad(image_bgr)
    img = torch.from_numpy(padded_bgr).to(device=device, dtype=dtype)  # (H, W, 3), BGR
    img = img.permute(2, 0, 1)[[2, 1, 0], :, :]  # -> (3, H, W), RGB
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(3, 1, 1)
    img = (img - mean) / std
    return img[None], padding_size, image_bgr.shape[:2]


def postprocess(raw_normal, padding_size, orig_shape):
    """raw_normal: (1, 3, 1024, 768) model output. Returns (H, W, 3) unit
    normal map at the original image resolution, camera/view-space per
    sapiens2's NORMAL.md."""
    pad_left, pad_right, pad_top, pad_bottom = padding_size
    h, w = raw_normal.shape[2], raw_normal.shape[3]
    cropped = raw_normal[:, :, pad_top : h - pad_bottom, pad_left : w - pad_right]
    resized = F.interpolate(cropped, size=orig_shape, mode="bilinear", align_corners=False)
    normal = resized / torch.norm(resized, dim=1, keepdim=True).clamp(min=1e-8)
    return normal[0].permute(1, 2, 0)  # (H, W, 3)


@torch.no_grad()
def estimate_normal(model, image_bgr, device="cuda", dtype=torch.float32):
    """Full pipeline: raw BGR uint8 image -> (H, W, 3) unit camera-space normal map."""
    inputs, padding_size, orig_shape = preprocess(image_bgr, device=device, dtype=dtype)
    raw_normal = model(inputs)
    return postprocess(raw_normal, padding_size, orig_shape)
