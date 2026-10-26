"""Shared paths, the Core ML conversion helper and reproducible-build fixes."""

import json
import uuid
from collections.abc import MutableMapping
from pathlib import Path

import coremltools as ct
import torch
from coremltools.proto import Model_pb2
from torch import nn

from modelconv.buildinfo import source_date

ROOT = Path(__file__).resolve().parents[4]
MODELS = ROOT / "models"
COREML_DIR = MODELS / "coreml"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalized(nn.Module):
    """Wrap a model so it takes RGB in [0, 1] and applies ImageNet mean/std inside the graph."""

    def __init__(self, model: nn.Module, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        """Store the wrapped model and the normalization constants."""
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> object:
        """Normalize then run the wrapped model."""
        return self.model((x - self.mean) / self.std)


def pin_metadata(metadata: MutableMapping[str, str]) -> None:
    """Replace the conversion timestamps coremltools and Ultralytics record with the source date.

    Args:
        metadata: The model's user-defined metadata.

    """
    date = source_date()
    metadata["com.github.apple.coremltools.conversion_date"] = date.date().isoformat()
    if "date" in metadata:  # written by the Ultralytics exporter
        metadata["date"] = date.isoformat()


def make_reproducible(package: Path) -> None:
    """Remove run-to-run differences: random item ids in Manifest.json and protobuf map ordering in model.mlmodel.

    Args:
        package: The saved mlpackage.

    """
    spec_path = package / "Data" / "com.apple.CoreML" / "model.mlmodel"
    spec = Model_pb2.Model()
    spec.ParseFromString(spec_path.read_bytes())
    spec_path.write_bytes(spec.SerializeToString(deterministic=True))
    path = package / "Manifest.json"
    manifest = json.loads(path.read_text())
    ids = {old: str(uuid.uuid5(uuid.NAMESPACE_URL, item["path"])).upper()
           for old, item in manifest["itemInfoEntries"].items()}
    # 根模型条目必须在最前，Core ML 编译器依赖这个顺序 / the root entry must come first for the Core ML compiler
    root = manifest["rootModelIdentifier"]
    entries = sorted(manifest["itemInfoEntries"].items(), key=lambda kv: (kv[0] != root, kv[1]["path"]))
    manifest["itemInfoEntries"] = {ids[old]: item for old, item in entries}
    manifest["rootModelIdentifier"] = ids[root]
    path.write_text(json.dumps(manifest, indent=4) + "\n")


def convert_traced(name: str, model: nn.Module, size: int, output_names: list[str]) -> Path:
    """Trace ``model`` at 1x3xSxS and save it as ``models/coreml/<name>_fp16.mlpackage``.

    The input is an RGB image scaled to [0, 1] by Core ML (ImageType scale=1/255).

    Args:
        name: Output model base name.
        model: Eval-mode module taking RGB in [0, 1].
        size: Square input side.
        output_names: Core ML output names, in model output order.

    Returns:
        Path of the saved mlpackage.

    """
    COREML_DIR.mkdir(parents=True, exist_ok=True)
    example = torch.rand(1, 3, size, size)
    with torch.no_grad():
        traced = torch.jit.trace(model.eval(), example, strict=False)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name=n) for n in output_names],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )
    pin_metadata(mlmodel.user_defined_metadata)
    out = COREML_DIR / f"{name}_fp16.mlpackage"
    mlmodel.save(str(out))
    make_reproducible(out)
    return out
