"""Fixture: miniature ward file with the real traps found on 11 Jul —
Jayanagar/Vijayanagar (substring trap), Bellanduru/Varthuru (transliteration),
and the alias-table wards (Kadugodi, Hagadur, HSR Layout, HAL Airport)."""
import json
from pathlib import Path

import pytest

WARDS = [
    "Jayanagar", "Jayanagar East", "Vijayanagar", "Bellanduru", "Varthuru",
    "HSR Layout", "Kadugodi", "Hagadur", "Garudachar Playa", "Hudi", "HAL Airport",
    "Yelahanka Satellite Town", "Chowdeswari Ward", "Atturu",
]


def _square(i: int) -> list:
    x, y = 77.5 + (i % 4) * 0.02, 12.9 + (i // 4) * 0.02
    return [[[x, y], [x + 0.02, y], [x + 0.02, y + 0.02], [x, y + 0.02], [x, y]]]


@pytest.fixture(scope="session")
def wards_file(tmp_path_factory) -> str:
    features = [
        {"type": "Feature",
         "properties": {"WARD_NAME": name, "WARD_NO": i + 1},
         "geometry": {"type": "Polygon", "coordinates": _square(i)}}
        for i, name in enumerate(WARDS)
    ]
    path = tmp_path_factory.mktemp("data") / "wards.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return str(path)
