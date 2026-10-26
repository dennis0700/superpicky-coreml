"""从 SuperPicky 的 eBird 地区库导出各国家、省州的鸟种名录（单独作为 regions 数据包发布）。

名录里是鸟种库类别号（0..10963，与 birdtrace_class_map.json、species_names.json 同一套编号），
App 识鸟时可只在拍摄地的名录里挑鸟种。输出为 JSON 数组，国家按代码排序，省州挂在所属国家下，
类别号升序；没有名录的国家和省州省略::

    [{"code": "CN", "en": "China", "zh": "中国", "species": [...],
      "regions": [{"code": "CN-35", "en": "Fujian", "zh": "福建省", "species": [...]}]}, ...]

只依赖标准库。用法（在仓库根目录）::

    uv run --project tools/modelconv python tools/export_regions.py \
        --db ../SuperPicky/birdid/data/ebird_regions.db \
        --out models/regions/regions.json

Export per-country and per-subnational species lists from SuperPicky's eBird region database (released as
the separate regions data package). The lists hold species-DB ids (0..10963, the same numbering as
birdtrace_class_map.json and species_names.json), so an app can pick species only from the shooting
region's list. Output is a JSON array of countries sorted by code, subnational regions nested under their
country, ids ascending; countries and regions without a list are omitted. Standard library only.
"""

import argparse
import json
import logging
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

LOGGER = logging.getLogger(__name__)
NUM_CLASSES = 10964


def load_regions(db_path: Path) -> list[dict[str, object]]:
    """读取地区库，返回国家及其省州的名录 / Read the region database: countries with their subnational lists.

    Args:
        db_path: ebird_regions.db 的路径 / path to ebird_regions.db.

    Returns:
        每个国家一条，省州在它的 "regions" 里 / one entry per country, subnational regions in its "regions".

    Raises:
        sqlite3.Error: 数据库无法打开或缺少 regions / region_species 表 / the database cannot be opened or lacks
            the regions / region_species tables.
        ValueError: 类别号超出 0..NUM_CLASSES-1 / a class id is outside 0..NUM_CLASSES-1.

    """
    # 只读打开，避免意外修改 SuperPicky 的数据库 / open read-only so SuperPicky's database is never modified
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT code, parent, name_en, name_zh FROM regions ORDER BY code").fetchall()
        species: dict[str, list[int]] = defaultdict(list)
        for region, class_id in connection.execute(
            "SELECT region, class_id FROM region_species ORDER BY region, class_id"
        ):
            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(f"{region}: class id {class_id} out of range")
            species[region].append(int(class_id))
    subregions: dict[str, list[dict[str, object]]] = defaultdict(list)
    for code, parent, en, zh in rows:
        if parent and species.get(code):
            subregions[parent].append({"code": code, "en": en, "zh": zh, "species": species[code]})
    return [
        {"code": code, "en": en, "zh": zh, "species": species[code], "regions": subregions[code]}
        for code, parent, en, zh in rows
        if not parent and species.get(code)
    ]


def main() -> int:
    """命令行入口 / Command-line entry point.

    Returns:
        进程退出码，0 表示成功 / process exit code, 0 on success.

    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, required=True, help="ebird_regions.db")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        countries = load_regions(args.db)
    except (sqlite3.Error, ValueError) as error:
        LOGGER.error("cannot read %s: %s", args.db, error)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(countries, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s (%d countries)", args.out, len(countries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
