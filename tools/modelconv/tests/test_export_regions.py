import sqlite3
import sys
from pathlib import Path

import pytest

# tools/ 下的脚本不在包里 / the scripts under tools/ are not part of the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from export_regions import load_regions  # noqa: E402


def _db(path: Path, species: list[tuple[str, int]]) -> Path:
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE regions (code TEXT, parent TEXT, name_en TEXT, name_zh TEXT, species_count INTEGER)")
    db.execute("CREATE TABLE region_species (region TEXT, class_id INTEGER)")
    db.executemany(
        "INSERT INTO regions VALUES (?, ?, ?, ?, 0)",
        [("CN", None, "China", "中国"), ("CN-35", "CN", "Fujian", "福建省"), ("CN-11", "CN", "Beijing", "北京市"),
         ("AQ", None, "Antarctica", "南极洲")],
    )
    db.executemany("INSERT INTO region_species VALUES (?, ?)", species)
    db.commit()
    db.close()
    return path


def test_subregions_nest_under_country_and_empty_lists_are_dropped(tmp_path: Path) -> None:
    path = _db(tmp_path / "ebird_regions.db", [("CN", 833), ("CN", 7), ("CN-35", 833)])
    assert load_regions(path) == [
        {"code": "CN", "en": "China", "zh": "中国", "species": [7, 833],
         "regions": [{"code": "CN-35", "en": "Fujian", "zh": "福建省", "species": [833]}]},
    ]


def test_out_of_range_class_id_is_rejected(tmp_path: Path) -> None:
    path = _db(tmp_path / "ebird_regions.db", [("CN", 10964)])
    with pytest.raises(ValueError):
        load_regions(path)
