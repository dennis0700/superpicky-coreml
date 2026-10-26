"""Export YOLO11l-seg to fp16 Core ML via Ultralytics (imgsz 640, nms=False).

The two outputs are renamed to ``detections`` (1x116x8400) and ``protos`` (1x32x160x160); letterboxing, NMS and
mask decoding happen in the app.
"""

import logging
import shutil

import coremltools as ct
from ultralytics import YOLO

from modelconv.common import COREML_DIR, MODELS, make_reproducible, pin_metadata

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Export, rename the outputs and move the package to models/coreml."""
    COREML_DIR.mkdir(parents=True, exist_ok=True)
    produced = YOLO(str(MODELS / "yolo11l-seg.pt")).export(format="coreml", imgsz=640, nms=False, quantize=16)
    dest = COREML_DIR / "yolo11l-seg_fp16.mlpackage"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(produced), str(dest))
    spec = ct.models.MLModel(str(dest), skip_model_load=True).get_spec()
    by_shape = {tuple(o.type.multiArrayType.shape): o.name for o in spec.description.output}
    ct.utils.rename_feature(spec, by_shape[(1, 116, 8400)], "detections")
    ct.utils.rename_feature(spec, by_shape[(1, 32, 160, 160)], "protos")
    pin_metadata(spec.description.metadata.userDefined)
    renamed = dest.with_suffix(".renamed.mlpackage")
    ct.models.MLModel(spec, weights_dir=str(dest / "Data/com.apple.CoreML/weights"), skip_model_load=True).save(
        str(renamed))
    shutil.rmtree(dest)
    shutil.move(str(renamed), str(dest))
    make_reproducible(dest)
    LOGGER.info("saved %s", dest)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
