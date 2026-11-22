# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


import tyro

from vhap_gnm.config.base import BaseTrackingConfig
from vhap_gnm.model.tracker import GlobalTracker


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    cfg = tyro.cli(BaseTrackingConfig)

    tracker = GlobalTracker(cfg)
    tracker.optimize()
