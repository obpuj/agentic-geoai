"""Tests for the numeric-grounding checker — the automatic half of Table 3."""
from auie.run_evaluation import grounding_check

EVIDENCE = ("2019: built_up 35.94 km2 (77.5%), vegetation 9.17 km2 (19.8%)\n"
            "built_up_change_km2: 2.5964\nbuilt_up_change_pct: 7.22")
CHUNKS = "mIoU 0.714; built-up 0.879"


def test_grounded_answer_passes():
    ans = ("Built-up grew from 35.94 km2 (77.5%) by 2.5964 km2 (7.22%), "
           "per a model with mIoU 0.714 in 2019.")
    ok, offenders = grounding_check(ans, EVIDENCE, CHUNKS)
    assert ok and offenders == []


def test_invented_number_caught():
    ans = "Built-up grew by 3.1 km2."
    ok, offenders = grounding_check(ans, EVIDENCE, CHUNKS)
    assert not ok and "3.1" in offenders


def test_rounded_copy_allowed_and_years_ignored():
    ans = "Roughly 2.6 km2 of growth between 2019 and 2025."
    ok, offenders = grounding_check(ans, EVIDENCE, CHUNKS)
    assert ok, offenders  # 2.6 is 2.5964 rounded; 2025 is a year

def test_sign_lexicalized_magnitude_passes():
    ev = "vegetation_change_km2: -3.2835"
    ok, offenders = grounding_check("Vegetation decreased by 3.2835 km2.", ev, "")
    assert ok, offenders
    