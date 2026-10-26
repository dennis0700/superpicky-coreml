"""把 models/coreml 下的模型连同许可证文件打成 dist/*.zip，并生成 dist/manifest.json（只依赖标准库）。

每个 zip 附带生成的 SOURCE.txt（日期、源码提交、上游权重）。同一个提交重复打包，结果逐字节相同。

用法（在仓库根目录）::

    python3 tools/package_models.py [--base-url https://…]
"""

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 上游权重的地址和 sha256 只在 fetch_models 里维护一份 / upstream URLs and hashes live only in fetch_models
sys.path.insert(0, str(ROOT / "tools" / "modelconv" / "src"))
from modelconv.buildinfo import source_date, source_version  # noqa: E402
from modelconv.fetch_models import WEIGHTS  # noqa: E402

LOGGER = logging.getLogger(__name__)
MODELS = ROOT / "models"
DIST = ROOT / "dist"
MANIFEST = DIST / "manifest.json"
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ModelSpec:
    """一个要发布的模型。

    Attributes:
        name: 目录名，同时是 manifest 里的模型名。
        package: models/coreml 下的 .mlpackage 目录名。
        extra_files: models/<name>/ 下需要一起打包的文件名。
        license_spdx: 许可证的 SPDX 标识。
        upstream: 上游项目地址。
        weights: 上游权重在 fetch_models.WEIGHTS 里的 dest，写进 SOURCE.txt。
        agpl_data: 附带了 AGPL-3.0 的数据文件，需要一起放 AGPL 许可证全文。

    """

    name: str
    package: str
    extra_files: tuple[str, ...]
    license_spdx: str
    upstream: str
    weights: str
    agpl_data: bool = False


SPECS = (
    ModelSpec(
        name="yolo11l-seg",
        package="yolo11l-seg_fp16.mlpackage",
        extra_files=("LICENSE", "NOTICE.txt"),
        license_spdx="AGPL-3.0-only",
        upstream="https://github.com/ultralytics/ultralytics",
        weights="yolo11l-seg.pt",
    ),
    ModelSpec(
        name="birdtrace",
        package="birdtrace_fp16.mlpackage",
        extra_files=("LICENSE", "NOTICE.txt", "THIRD_PARTY_NOTICES.txt", "birdtrace_class_map.json",
                     "species_names.json"),
        license_spdx="CC-BY-NC-4.0",
        upstream="https://huggingface.co/warden123/BirdTrace-FastViT-MA36-10K",
        weights="birdtrace/model.safetensors",
        agpl_data=True,
    ),
    ModelSpec(
        name="topiq",
        package="topiq_fp16.mlpackage",
        extra_files=("LICENSE", "NOTICE.txt"),
        license_spdx="CC-BY-NC-SA-4.0",
        upstream="https://github.com/chaofengc/IQA-PyTorch",
        weights="cfanet_iaa_ava_res50_fp16.pth",
    ),
    # GPL-3.0 依据 SuperPicky-models 的模型卡 / per the SuperPicky-models model card
    ModelSpec(
        name="keypoint",
        package="keypoint_fp16.mlpackage",
        extra_files=("LICENSE", "NOTICE.txt"),
        license_spdx="GPL-3.0-only",
        upstream="https://github.com/jamesphotography/SuperPicky",
        weights="cub200_keypoint_resnet50_slim.pth",
    ),
    ModelSpec(
        name="flight",
        package="flight_fp16.mlpackage",
        extra_files=("LICENSE", "NOTICE.txt"),
        license_spdx="GPL-3.0-only",
        upstream="https://github.com/jamesphotography/SuperPicky",
        weights="superFlier_efficientnet.pth",
    ),
)


def sha256_of(path: Path) -> str:
    """计算文件的 sha256。

    Args:
        path: 要计算的文件。

    Returns:
        小写十六进制摘要。

    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def source_notes(spec: ModelSpec, version: str, date: datetime) -> str:
    """生成 SOURCE.txt：日期、转换源码的版本和上游权重。

    Args:
        spec: 模型描述。
        version: `source_version()` 的结果。
        date: 源码日期，即 GPL/AGPL 第 5(a) 条要求的修改日期。

    Returns:
        SOURCE.txt 的内容。

    """
    weight = next(w for w in WEIGHTS if w.dest == spec.weights)
    return (f"{spec.package}: how this package was made\n\n"
            f"Date:               {date.date().isoformat()}\n"
            f"Conversion source:  superpicky-coreml, {version}\n"
            f"                    tools/modelconv (fetch_models, then convert_all; see README.md)\n"
            f"Upstream weights:   {weight.url}\n"
            f"                    sha256 {weight.sha256}\n")


def build_zip(spec: ModelSpec, version: str, date: datetime) -> Path:
    """把一个模型的 mlpackage 与许可证文件打成 zip，文件顺序和时间固定。

    Args:
        spec: 模型描述。
        version: 转换源码的版本，写进 SOURCE.txt。
        date: 源码日期，用作 zip 内所有文件的时间。

    Returns:
        生成的 zip 路径。

    Raises:
        FileNotFoundError: mlpackage 或任一附带文件不存在。

    """
    package = MODELS / "coreml" / spec.package
    if not package.is_dir():
        raise FileNotFoundError(f"{package} 不存在，请先运行 modelconv.convert_all")
    files = [(path.read_bytes(), str(Path(spec.package) / path.relative_to(package)))
             for path in sorted(p for p in package.rglob("*") if p.is_file())]
    files += [((MODELS / spec.name / name).read_bytes(), name) for name in spec.extra_files]
    if spec.agpl_data:
        files.append(((ROOT / "LICENSE").read_bytes(), "LICENSE-AGPL-3.0.txt"))
    files.append((source_notes(spec, version, date).encode(), "SOURCE.txt"))
    DIST.mkdir(exist_ok=True)
    target = DIST / f"{spec.name}-fp16.zip"
    stamp = max(date.timetuple()[:6], (1980, 1, 1, 0, 0, 0))   # zip 最早只能表示 1980 年 / zip starts at 1980
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for data, name in files:
            info = zipfile.ZipInfo(name, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return target


def build_entry(spec: ModelSpec, archive: Path, base_url: str | None) -> dict[str, object]:
    """生成 manifest 里的一条记录。

    Args:
        spec: 模型描述。
        archive: 已打好的 zip。
        base_url: 下载地址前缀；为 None 时不写 url。

    Returns:
        可直接序列化为 JSON 的字典。

    """
    entry: dict[str, object] = {
        "name": spec.name,
        "file": archive.name,
        "package": spec.package,
        "size": archive.stat().st_size,
        "sha256": sha256_of(archive),
        "license": spec.license_spdx,
        "upstream": spec.upstream,
    }
    if base_url:
        entry["url"] = f"{base_url.rstrip('/')}/{archive.name}"
    return entry


def main() -> int:
    """命令行入口。

    Returns:
        进程退出码，0 表示成功。

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="下载地址前缀，写入 manifest 的 url 字段")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    version, date = source_version(), source_date()
    if "uncommitted" in version:
        LOGGER.warning("working tree has uncommitted changes; these packages are not for release")
    entries = []
    for spec in SPECS:
        archive = build_zip(spec, version, date)
        LOGGER.info("packaged %s (%d bytes)", archive.name, archive.stat().st_size)
        entries.append(build_entry(spec, archive, args.base_url))
    text = json.dumps({"models": entries}, ensure_ascii=False, indent=2) + "\n"
    MANIFEST.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
