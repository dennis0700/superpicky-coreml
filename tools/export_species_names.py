"""从 SuperPicky 的鸟种库导出鸟种名称表（随 birdtrace 模型发布）。

BirdTrace 的输出经 birdtrace_class_map.json 映射到鸟种库类别号（0..10963，沿用 OSEA 编号），
本脚本把每个类别号对应的简体中文名、英文名、学名导出为 JSON 数组（下标即类别号），
供 App 显示中文鸟种名。只依赖标准库。

用法（在仓库根目录）::

    python3 tools/export_species_names.py \
        --db ../SuperPicky/birdid/data/bird_reference.sqlite \
        --out models/birdtrace/species_names.json

Export the species name table (shipped with the birdtrace model) from SuperPicky's species database.
BirdTrace outputs are mapped through birdtrace_class_map.json onto species-DB ids (0..10963, OSEA
numbering). This script writes, per id, the Simplified Chinese, English and scientific names as a JSON
array whose index is the id. Standard library only.
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)
NUM_CLASSES = 10964


def load_names(db_path: Path) -> list[dict[str, str]]:
    """读取鸟种库，返回按类别号排列的名称列表 / Read the species database, names ordered by class id.

    Args:
        db_path: bird_reference.sqlite 的路径 / path to bird_reference.sqlite.

    Returns:
        长度为 NUM_CLASSES 的列表，每项含 zh / en / sci，缺失的类别为空字符串 / NUM_CLASSES entries with
        zh / en / sci; missing ids get empty strings.

    Raises:
        sqlite3.Error: 数据库无法打开或缺少 BirdCountInfo 表 / the database cannot be opened or has no
            BirdCountInfo table.

    """
    names = [{"zh": "", "en": "", "sci": ""} for _ in range(NUM_CLASSES)]
    # 只读打开，避免意外修改 SuperPicky 的数据库 / open read-only so SuperPicky's database is never modified
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT model_class_id, chinese_simplified, english_name, scientific_name "
            "FROM BirdCountInfo WHERE model_class_id IS NOT NULL"
        )
        for class_id, zh, en, sci in rows:
            if 0 <= class_id < NUM_CLASSES:
                names[class_id] = {"zh": zh or "", "en": en or "", "sci": sci or ""}
    return names


def main() -> int:
    """命令行入口 / Command-line entry point.

    Returns:
        进程退出码，0 表示成功 / process exit code, 0 on success.

    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, required=True, help="bird_reference.sqlite")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        names = load_names(args.db)
    except sqlite3.Error as error:
        LOGGER.error("cannot read %s: %s", args.db, error)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(names, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    filled = sum(1 for entry in names if entry["zh"])
    LOGGER.info("wrote %s (%d/%d with Chinese names)", args.out, filled, NUM_CLASSES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
