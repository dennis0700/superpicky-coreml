"""带哈希校验的下载工具（仅依赖标准库）。"""

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path

LOGGER = logging.getLogger(__name__)
USER_AGENT = "superpicky-coreml/0.1 (Core ML model conversion asset fetcher) python-urllib"
CHUNK_SIZE = 1024 * 1024


class ChecksumError(RuntimeError):
    """文件哈希与期望值不一致。"""


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """计算文件哈希。

    Args:
        path: 文件路径。
        algorithm: hashlib 支持的算法名，例如 "sha256"、"sha1"。

    Returns:
        十六进制小写哈希字符串。

    """
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: dict[str, str]) -> None:
    """按多个算法校验文件哈希。

    Args:
        path: 文件路径。
        expected: 算法名到期望哈希的映射，空值会被跳过。

    Raises:
        ChecksumError: 任一哈希不匹配。

    """
    for algorithm, value in expected.items():
        if not value:
            continue
        actual = file_digest(path, algorithm)
        if actual != value.lower():
            raise ChecksumError(f"{path}: {algorithm} {actual} != expected {value}")


def fetch(url: str, dest: Path, expected: dict[str, str]) -> Path:
    """下载文件：已存在且校验通过则跳过，否则下载后校验。

    下载先写入 ``dest`` 同目录的 ``.part`` 临时文件，校验通过后再重命名，避免留下半截文件。

    Args:
        url: 下载地址。
        dest: 目标路径。
        expected: 算法名到期望哈希的映射。

    Returns:
        目标文件路径。

    Raises:
        ChecksumError: 获取到的文件哈希不匹配。
        urllib.error.URLError: 网络请求失败。

    """
    if dest.exists():
        verify(dest, expected)
        LOGGER.info("ok (cached) %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    LOGGER.info("download %s -> %s", url, dest)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out, CHUNK_SIZE)
    try:
        verify(part, expected)
    except ChecksumError:
        part.unlink()
        raise
    part.rename(dest)
    return dest
