import pytest

from modelconv.common import pin_metadata


def test_metadata_dates_do_not_follow_the_commit_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1893456000")  # 2030-01-01
    metadata = {"com.github.apple.coremltools.conversion_date": "2031-05-05", "date": "2031-05-05T00:00:00"}
    pin_metadata(metadata)
    assert metadata == {"com.github.apple.coremltools.conversion_date": "2026-10-26",
                        "date": "2026-10-26T10:24:32+00:00"}


def test_metadata_without_ultralytics_date_gets_no_date_key() -> None:
    metadata: dict[str, str] = {}
    pin_metadata(metadata)
    assert metadata == {"com.github.apple.coremltools.conversion_date": "2026-10-26"}
