"""构建日期和源码版本（只依赖标准库）/ Build date and source version (standard library only)."""

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _git(*args: str) -> str | None:
    """在仓库根目录运行 git，失败时返回 None。"""
    try:
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def source_date() -> datetime:
    """构建日期：优先 SOURCE_DATE_EPOCH，其次当前提交的时间，都没有时用现在。

    Returns:
        UTC 时间。

    """
    epoch = os.environ.get("SOURCE_DATE_EPOCH") or _git("log", "-1", "--format=%ct")
    return datetime.fromtimestamp(int(epoch), UTC) if epoch else datetime.now(UTC)


def source_version() -> str:
    """当前提交；有未提交的改动时注明（这样构建的产物不能用于发布）。

    Returns:
        描述版本的一行文字。

    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return "unknown (not a git checkout)"
    dirty = _git("status", "--porcelain")
    return f"commit {commit}" + (" plus uncommitted changes (not a release build)" if dirty else "")
