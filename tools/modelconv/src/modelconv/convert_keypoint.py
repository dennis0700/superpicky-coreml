"""Convert SuperEyes keypoint ResNet50 (PartLocalizer, 416x416) to fp16 Core ML.

ImageNet mean/std are baked into the graph; the app resizes the bird crop directly to 416x416. Outputs: ``coords``
(1x3x2, 0-1, [left eye, right eye, beak]) and ``vis`` (1x3).
"""

import logging

import torch

from modelconv.archs import PartLocalizer
from modelconv.common import IMAGENET_MEAN, IMAGENET_STD, MODELS, Normalized, convert_traced

LOGGER = logging.getLogger(__name__)
INPUT = 416


def load_keypoint() -> PartLocalizer:
    """Load cub200_keypoint_resnet50_slim.pth into PartLocalizer (eval, fp32)."""
    model = PartLocalizer()
    ckpt = torch.load(MODELS / "cub200_keypoint_resnet50_slim.pth", map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict({k: v.float() for k, v in ckpt.items()})
    return model.eval()


def main() -> None:
    """Convert to models/coreml/keypoint_fp16.mlpackage."""
    model = Normalized(load_keypoint(), IMAGENET_MEAN, IMAGENET_STD)
    LOGGER.info("saved %s", convert_traced("keypoint", model, INPUT, ["coords", "vis"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
