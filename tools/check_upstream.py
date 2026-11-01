"""检查 Hugging Face 上的上游权重是否和 fetch_models 里固定的 sha256 一致（只依赖标准库）。

不下载权重，只查 HF 的文件列表（LFS 文件的 oid 就是 sha256）。每个变化的权重输出一行 Markdown 列表项，
都没变时没有输出；查询失败时以非 0 退出。SuperPicky 的数据由 check-upstream workflow 重新导出后 git diff 判断。

用法（在仓库根目录）::

    uv run --no-project python tools/check_upstream.py
"""

import json
import sys
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 上游权重的地址和 sha256 只在 fetch_models 里维护一份 / upstream URLs and hashes live only in fetch_models
sys.path.insert(0, str(ROOT / "tools" / "modelconv" / "src"))
from modelconv.download import USER_AGENT  # noqa: E402
from modelconv.fetch_models import HF, WEIGHTS, Weight  # noqa: E402


def lfs_sha256s(repo: str) -> dict[str, str]:
    """列出 HF 仓库 main 分支根目录下 LFS 文件的 sha256。

    Args:
        repo: HF 仓库名，例如 "Ultralytics/YOLO11"。

    Returns:
        文件名到 sha256 的映射。

    Raises:
        urllib.error.URLError: 网络请求失败。

    """
    request = urllib.request.Request(f"{HF}/api/models/{repo}/tree/main", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        entries = json.load(response)
    return {entry["path"]: entry["lfs"]["oid"] for entry in entries if "lfs" in entry}


def changed_weights(weights: Iterable[Weight], lookup: Callable[[str], dict[str, str]]) -> list[str]:
    """找出上游 sha256 和固定值不一致（或上游已没有该文件）的权重。

    Args:
        weights: 要检查的权重。
        lookup: 仓库名到「文件名 -> sha256」的查询函数，每个仓库只调用一次。

    Returns:
        每个变化的权重一行 Markdown 列表项。

    """
    cache: dict[str, dict[str, str]] = {}
    lines = []
    for weight in weights:
        if weight.repo not in cache:
            cache[weight.repo] = lookup(weight.repo)
        upstream = cache[weight.repo].get(weight.filename)
        if upstream != weight.sha256:
            now = f"upstream `{upstream}`" if upstream else "no longer upstream"
            lines.append(f"- `{weight.dest}` ({weight.repo}/{weight.filename}): pinned `{weight.sha256}`, {now}")
    return lines


def main() -> int:
    """命令行入口。

    Returns:
        进程退出码，0 表示查询成功（不论有没有变化）。

    """
    for line in changed_weights(WEIGHTS, lfs_sha256s):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
