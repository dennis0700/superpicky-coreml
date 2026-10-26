"""Convert TOPIQ CFANet (AVA aesthetic, ResNet50) to fp16 Core ML at a fixed 384x384 input.

Starts from SuperPicky's fp16 copy of the upstream weights. CFANet normalizes internally; the app resizes the bird
crop to 384x384 (Lanczos). Output: ``mos``, the aesthetic score (1-10).
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from modelconv.archs import CFANet, clean_state_dict
from modelconv.common import MODELS, convert_traced

LOGGER = logging.getLogger(__name__)
INPUT = 384
WEIGHTS = "cfanet_iaa_ava_res50_fp16.pth"


def load_topiq() -> CFANet:
    """Load the CFANet checkpoint as SuperPicky's load_topiq_weights does (strict=False, reported)."""
    model = CFANet()
    state = torch.load(MODELS / WEIGHTS, map_location="cpu", weights_only=True)
    if "params" in state:
        state = state["params"]
    state = {k: (v.float() if v.is_floating_point() else v) for k, v in clean_state_dict(state).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    LOGGER.info("%s missing=%d unexpected=%d", WEIGHTS, len(missing), len(unexpected))
    return model.eval()


@contextmanager
def static_bicubic(model: CFANet) -> Iterator[None]:
    """Replace F.interpolate(mode='bicubic') with constants precomputed for the fixed 384x384 input.

    Core ML has no ``upsample_bicubic2d`` converter. At a fixed input size the positional embedding
    resize only depends on parameters, so it can be evaluated eagerly (same PyTorch bicubic math)
    and traced as a constant. Valid only for INPUT x INPUT.
    """
    original = torch.nn.functional.interpolate
    with torch.no_grad():
        pos = torch.cat((model.h_emb.repeat(1, 1, 1, model.w_emb.shape[3]),
                         model.w_emb.repeat(1, 1, model.h_emb.shape[2], 1)), dim=1)
        # every level is adaptive-avg-pooled to the stride-32 grid (12x12 at 384) before the resize
        side = INPUT // 32
        constant = original(pos, size=(side, side), mode="bicubic", align_corners=False).clone()

    def patched(x: torch.Tensor, size: object = None, **kwargs: object) -> torch.Tensor:
        if kwargs.get("mode") == "bicubic" and (int(size[0]), int(size[1])) == (side, side):
            return constant
        return original(x, size=size, **kwargs)

    torch.nn.functional.interpolate = patched
    try:
        yield
    finally:
        torch.nn.functional.interpolate = original


def main() -> None:
    """Convert to models/coreml/topiq_fp16.mlpackage."""
    model = load_topiq()
    with static_bicubic(model):
        LOGGER.info("saved %s", convert_traced("topiq", model, INPUT, ["mos"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
