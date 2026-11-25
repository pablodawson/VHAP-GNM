# Offline preprocessing pass that annotates each frame of a sequence with a
# per-pixel camera-space surface normal map, predicted by Sapiens2
# (facebookresearch/sapiens2). Mirrors the caching pattern of
# landmark_detector_star.py, but runs sequentially in a single process: this
# is a multi-hundred-million-parameter GPU model, not something to fork
# across CPU workers the way STAR/dlib are.

import os
import cv2
from collections import Counter
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from vhap_gnm.util.log import get_logger
from vhap_gnm.util.sapiens2_normal import load_sapiens2_normal, estimate_normal

logger = get_logger(__name__)


class NormalDetectorSapiens2:
    def __init__(self, checkpoint_path, arch="sapiens2_0.4b", device="cuda"):
        logger.info(f"Initialize Normal Detector (Sapiens2 {arch})...")
        self.model = load_sapiens2_normal(checkpoint_path, arch=arch, device=device)
        self.device = device

    def detect_single_image(self, img_rgb):
        """img_rgb: (H, W, 3) uint8, RGB order (as returned by PIL.Image.open).
        Returns (H, W, 3) float32 unit camera-space normal map."""
        img_bgr = img_rgb[:, :, ::-1]
        normal = estimate_normal(self.model, img_bgr, device=self.device)
        return normal.cpu().numpy().astype(np.float32)


def annotate_normals(dataset, checkpoint_path, arch="sapiens2_0.4b", device="cuda"):
    """
    Annotates each frame with a Sapiens2 camera-space normal map, caching one
    .npy file per frame (mirroring how alpha_maps are stored per-timestep).
    """
    os.umask(0o002)
    detector = NormalDetectorSapiens2(checkpoint_path, arch=arch, device=device)

    # a run killed mid-write can leave a truncated .npy, which the skip-if-exists below would keep forever:
    # every normal map has the same shape, so drop files whose size differs from the common one
    cache_dir = dataset.get_property_path("normal", 0).parent
    if cache_dir.exists():
        for tmp in cache_dir.glob("*.tmp.npy"):
            tmp.unlink()
        sizes = {p: p.stat().st_size for p in cache_dir.glob("*.npy")}
        if len(sizes) > 1:
            common = Counter(sizes.values()).most_common(1)[0][0]
            for p, size in sizes.items():
                if size != common:
                    logger.warning(f"Removing truncated normal map {p} ({size} bytes, expected {common})")
                    p.unlink()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    for item in tqdm(dataloader, desc="Sapiens2 normals"):
        timestep_id = item["timestep_id"][0]
        camera_id = item["camera_id"][0]
        index = item["timestep_index"][0].item()

        out_path = dataset.get_property_path("normal", index)
        if out_path.exists():
            continue

        logger.info(f"Annotate normal map for timestep: {timestep_id}, camera: {camera_id}")
        img = item["rgb"][0].numpy()
        normal = detector.detect_single_image(img)

        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        # float16 halves the cache (~2.3 MB/frame at 640x608); loader casts to float32. Written atomically.
        tmp = out_path.with_name(out_path.stem + ".tmp.npy")
        np.save(tmp, normal.astype(np.float16))
        os.replace(tmp, out_path)


if __name__ == "__main__":
    import tyro
    from vhap_gnm.config.base import DataConfig, import_module

    cfg = tyro.cli(DataConfig)
    dataset = import_module(cfg._target)(cfg=cfg, img_to_tensor=False)
    annotate_normals(dataset, cfg.normal_checkpoint, arch=cfg.normal_arch, device=cfg.normal_device)
