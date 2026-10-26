"""Convert all five models to fp16 Core ML; stops at the first failure."""

import logging

from modelconv import (
    convert_birdtrace,
    convert_flight,
    convert_keypoint,
    convert_topiq,
    convert_yolo,
)

LOGGER = logging.getLogger(__name__)
STEPS = {
    "yolo": convert_yolo.main,
    "birdtrace": convert_birdtrace.main,
    "keypoint": convert_keypoint.main,
    "flight": convert_flight.main,
    "topiq": convert_topiq.main,
}


def main() -> None:
    """Run every converter in turn."""
    for name, convert in STEPS.items():
        LOGGER.info("== converting %s", name)
        convert()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
