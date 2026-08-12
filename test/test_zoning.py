import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from volume_finder.core import zoning


def test_resolve_zone_matching_code_and_name():
    key, warning = zoning.resolve_zone(5, "第一種住居地域")
    assert key == "1住"
    assert warning is None


def test_resolve_zone_accepts_arabic_numeral_name_variant():
    key, warning = zoning.resolve_zone(5, "第1種住居地域")
    assert key == "1住"
    assert warning is None


def test_resolve_zone_prefers_name_on_code_mismatch():
    # code says 準工(10) but name says 商業 -- code系年次のずれを想定
    key, warning = zoning.resolve_zone(10, "商業地域")
    assert key == "商業"
    assert warning is not None
    assert "名称を優先" in warning


def test_resolve_zone_falls_back_to_code_when_name_unrecognized():
    key, warning = zoning.resolve_zone(9, "謎の地域名")
    assert key == "商業"
    assert "コード" in warning


def test_resolve_zone_none_when_both_unrecognized():
    key, warning = zoning.resolve_zone(None, "謎の地域名")
    assert key is None
    assert "手動選択" in warning


def test_prorate_weighted_average():
    zones = [
        {"far": 200, "bcr": 60, "area": 2100.5},
        {"far": 200, "bcr": 60, "area": 1649.7},
    ]
    total = 2100.5 + 1649.7
    far, bcr = zoning.prorate(zones, total)
    assert math.isclose(far, 200.0)
    assert math.isclose(bcr, 60.0)


def test_prorate_differing_far():
    zones = [{"far": 200, "bcr": 60, "area": 100}, {"far": 300, "bcr": 60, "area": 100}]
    far, bcr = zoning.prorate(zones, 200)
    assert math.isclose(far, 250.0)


def test_prorate_undercoverage_dilutes_result():
    zones = [{"far": 400, "bcr": 60, "area": 50}]
    far, _ = zoning.prorate(zones, 100)  # only half the site covered
    assert math.isclose(far, 200.0)


def test_boundary_warning_near_step():
    assert zoning.boundary_warning(198.5) is not None
    assert zoning.boundary_warning(250.0) is None


def test_coverage_warning():
    assert zoning.coverage_warning(90, 100) is not None
    assert zoning.coverage_warning(99, 100) is None
