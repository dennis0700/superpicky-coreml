import sys
from pathlib import Path

# tools/ 下的脚本不在包里 / the scripts under tools/ are not part of the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from check_upstream import changed_weights  # noqa: E402

from modelconv.fetch_models import Weight

WEIGHTS = (
    Weight("a.pt", "org/one", "a.pt", "aaa"),
    Weight("b.pt", "org/one", "b.pt", "bbb"),
    Weight("c.pt", "org/two", "c.pt", "ccc"),
)


def test_unchanged_weights_report_nothing_and_each_repo_is_queried_once() -> None:
    calls = []

    def lookup(repo: str) -> dict[str, str]:
        calls.append(repo)
        return {"a.pt": "aaa", "b.pt": "bbb", "c.pt": "ccc", "other.pt": "zzz"}

    assert changed_weights(WEIGHTS, lookup) == []
    assert calls == ["org/one", "org/two"]


def test_changed_and_removed_weights_are_reported() -> None:
    upstream = {"org/one": {"a.pt": "aaa", "b.pt": "new"}, "org/two": {}}
    lines = changed_weights(WEIGHTS, upstream.__getitem__)
    assert len(lines) == 2
    assert "`b.pt`" in lines[0] and "upstream `new`" in lines[0]
    assert "`c.pt`" in lines[1] and "no longer upstream" in lines[1]
