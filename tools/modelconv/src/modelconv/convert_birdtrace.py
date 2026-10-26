"""Convert BirdTrace FastViT-MA36 (384x384) to fp16 Core ML.

ImageNet mean/std are baked into the graph; the output is raw logits (10480), softmax and the class map are applied
in the app.
"""

import json
import logging

import timm
from safetensors.torch import load_file
from timm.models.fastvit import checkpoint_filter_fn
from timm.utils.model import reparameterize_model
from torch import nn

from modelconv.common import IMAGENET_MEAN, IMAGENET_STD, MODELS, Normalized, convert_traced

LOGGER = logging.getLogger(__name__)
INPUT = 384


def load_birdtrace() -> nn.Module:
    """Build timm fastvit_ma36, load released weights strictly and reparameterize (eval mode)."""
    class_map = json.loads((MODELS / "birdtrace" / "birdtrace_class_map.json").read_text())["class_map"]
    state = load_file(str(MODELS / "birdtrace" / "model.safetensors"), device="cpu")
    state = {k.removeprefix("backbone."): v for k, v in state.items()}
    model = timm.create_model("fastvit_ma36", pretrained=False, num_classes=len(class_map))
    model.load_state_dict(checkpoint_filter_fn(state, model), strict=True)
    return reparameterize_model(model.eval())


def main() -> None:
    """Convert to models/coreml/birdtrace_fp16.mlpackage."""
    LOGGER.info("saved %s", convert_traced("birdtrace", Normalized(load_birdtrace(), IMAGENET_MEAN, IMAGENET_STD),
                                           INPUT, ["logits"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
