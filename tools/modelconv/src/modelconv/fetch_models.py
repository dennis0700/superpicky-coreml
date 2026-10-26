"""下载转换所需的上游权重到 models/ 并校验 sha256。

用法（在 superpicky-coreml/ 根目录）::

    uv run --project tools/modelconv python -m modelconv.fetch_models
"""

import argparse
import logging
import sys
import urllib.error
from dataclasses import dataclass
from pathlib import Path

from modelconv.download import ChecksumError, fetch

REPO_ROOT = Path(__file__).resolve().parents[4]
HF = "https://huggingface.co"


@dataclass(frozen=True)
class Weight:
    """一个待下载的模型文件。"""

    dest: str
    repo: str
    filename: str
    sha256: str

    @property
    def url(self) -> str:
        """Hugging Face resolve 下载地址。"""
        return f"{HF}/{self.repo}/resolve/main/{self.filename}"


SP_MODELS = "jamesphotography/SuperPicky-models"

# sha256 来源：SuperPicky scripts/download_models.py、BirdTrace SHA256SUMS、HF LFS oid。
WEIGHTS: tuple[Weight, ...] = (
    Weight(
        "yolo11l-seg.pt",
        "Ultralytics/YOLO11",
        "yolo11l-seg.pt",
        "cabe90049795dfc9a370b7934d6dec7f6b9e44a20e573b0ff81b7e205512c872",
    ),
    Weight(
        "birdtrace/model.safetensors",
        "warden123/BirdTrace-FastViT-MA36-10K",
        "model.safetensors",
        "81d698a8457a9c40d67e6d20d18ebe24f9d164b7083717b9f18a08a99f9f0ca6",
    ),
    Weight(
        "cub200_keypoint_resnet50_slim.pth",
        SP_MODELS,
        "cub200_keypoint_resnet50_slim.pth",
        "25bee11a4846e9065185bf22512e0ce4f81a62da79aece8f1683209f375904a6",
    ),
    Weight(
        "superFlier_efficientnet.pth",
        SP_MODELS,
        "superFlier_efficientnet.pth",
        "cf2d5bd10fff0af83fbf57fe365221339152aedd712c4bbbb5b757f2838451a7",
    ),
    Weight(
        "cfanet_iaa_ava_res50_fp16.pth",
        SP_MODELS,
        "cfanet_iaa_ava_res50_fp16.pth",
        "db9a31eae686b3cd82200d5e2e2b040252c102c225bf18ff22c83f00cfe9b885",
    ),
)


def fetch_all(models_dir: Path) -> list[str]:
    """下载全部权重，单个失败不影响其余文件。

    Args:
        models_dir: 目标目录（superpicky-coreml/models）。

    Returns:
        失败条目的描述列表；为空表示全部成功。

    """
    failures: list[str] = []
    for weight in WEIGHTS:
        try:
            fetch(weight.url, models_dir / weight.dest, {"sha256": weight.sha256})
        except (urllib.error.URLError, TimeoutError, ChecksumError, OSError) as error:
            logging.error("failed %s: %s", weight.dest, error)
            failures.append(f"{weight.dest}: {error}")
    return failures


def main() -> int:
    """命令行入口。

    Returns:
        进程退出码，0 表示全部成功。

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    failures = fetch_all(args.models_dir)
    for failure in failures:
        print(f"MISSING {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
