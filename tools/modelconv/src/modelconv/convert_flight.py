"""Convert SuperFlier EfficientNet-B3 flight classifier (384x384) to fp16 Core ML.

ImageNet mean/std are baked into the graph; the app resizes the bird crop directly to 384x384. Output: ``prob``,
the flight probability (sigmoid already in the graph).
"""

import logging

import torch

from modelconv.archs import build_flight_model
from modelconv.common import IMAGENET_MEAN, IMAGENET_STD, MODELS, Normalized, convert_traced

LOGGER = logging.getLogger(__name__)
INPUT = 384


def load_flight() -> torch.nn.Module:
    """Load superFlier_efficientnet.pth (eval, fp32)."""
    model = build_flight_model()
    state = torch.load(MODELS / "superFlier_efficientnet.pth", map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return model.eval()


def main() -> None:
    """Convert to models/coreml/flight_fp16.mlpackage."""
    model = Normalized(load_flight(), IMAGENET_MEAN, IMAGENET_STD)
    LOGGER.info("saved %s", convert_traced("flight", model, INPUT, ["prob"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
