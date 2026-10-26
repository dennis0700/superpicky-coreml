import sys
import zipfile
from pathlib import Path

# tools/ 下的脚本不在包里 / the scripts under tools/ are not part of the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from package_models import content_sha256  # noqa: E402


def _zip(path: Path, files: list[tuple[str, bytes]], date_time: tuple[int, ...] = (2026, 1, 1, 0, 0, 0)) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files:
            archive.writestr(zipfile.ZipInfo(name, date_time=date_time), data)
    return path


def test_content_hash_ignores_source_notes_order_and_dates(tmp_path: Path) -> None:
    a = _zip(tmp_path / "a.zip", [("m/w.bin", b"weights"), ("LICENSE", b"l"), ("SOURCE.txt", b"commit 1")])
    b = _zip(tmp_path / "b.zip", [("LICENSE", b"l"), ("SOURCE.txt", b"commit 2"), ("m/w.bin", b"weights")],
             date_time=(2027, 6, 1, 0, 0, 0))
    assert content_sha256(a) == content_sha256(b)


def test_content_hash_changes_with_any_other_file(tmp_path: Path) -> None:
    a = _zip(tmp_path / "a.zip", [("m/w.bin", b"weights"), ("LICENSE", b"l")])
    b = _zip(tmp_path / "b.zip", [("m/w.bin", b"weights2"), ("LICENSE", b"l")])
    c = _zip(tmp_path / "c.zip", [("m/w.bin", b"weights"), ("NOTICE", b"l")])
    assert len({content_sha256(a), content_sha256(b), content_sha256(c)}) == 3
