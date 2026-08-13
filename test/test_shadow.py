import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from volume_finder.core import geometry as geo
from volume_finder.core import regulation as reg
from volume_finder.core import shadow


def test_sun_position_matches_design_reference_at_8am():
    # 設計書 8.1: 北緯35度・冬至日、8時の方位角 −53°27′、倍率 6.71
    az, mult = shadow.sun_position(8.0, phi_deg=35.0)
    az_deg = math.degrees(az)
    expected_deg = -(53 + 27 / 60)
    assert math.isclose(az_deg, expected_deg, abs_tol=0.05)
    assert math.isclose(mult, 6.71, abs_tol=0.02)


def test_sun_position_none_when_sun_below_horizon():
    assert shadow.sun_position(4.0, phi_deg=35.0) is None
    assert shadow.sun_position(20.0, phi_deg=35.0) is None


def test_sun_position_noon_azimuth_near_zero_south():
    az, mult = shadow.sun_position(12.0, phi_deg=35.0)
    assert math.isclose(az, 0.0, abs_tol=1e-6)
    assert mult > 0


def test_shadow_bands_zero_when_building_below_measurement_height():
    boundary = [(-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0)]
    bands = shadow.shadow_bands(0, 0, 20, 20, 0, H=3.0, deemed_boundary=boundary, mh=4.0)
    assert bands["max_5_10"] == 0.0
    assert bands["max_10plus"] == 0.0


def test_shadow_bands_taller_building_casts_more_shadow():
    # 同じグリッド（cell固定）で比較しないと、H依存で自動決まる余白/解像度が
    # short/tallで変わってしまい、離散サンプリングの都合で局所的に
    # 逆転することがある（連続場としては各時刻・各点で単調増加のはず:
    # offsetの大きさ=eff_top×倍率 はHが大きいほど大きく、向きは同じなので
    # 「箱+オフセット線分」のミンコフスキー和は時刻ごとにHが大きいほうを
    # 包含する）。cellを固定してグリッドを揃えることで単調性を検証する。
    boundary = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)]
    short = shadow.shadow_bands(0, 0, 10, 10, 0, H=10.0, deemed_boundary=boundary, mh=4.0, dt_minutes=10, cell=1.0)
    tall = shadow.shadow_bands(0, 0, 10, 10, 0, H=40.0, deemed_boundary=boundary, mh=4.0, dt_minutes=10, cell=1.0)
    assert tall["max_10plus"] >= short["max_10plus"]
    assert tall["max_5_10"] >= short["max_5_10"] - 1e-9


def test_shadow_bands_negligible_effective_height_casts_no_far_shadow():
    # 冬至日8時付近は倍率が6〜7倍と大きいため、有効高さ(H-mh)がわずかでも
    # 影は数m伸びる。有効高さをごく小さくして、境界(30m)の外・10m超帯まで
    # 届かないことを確認する。
    boundary = [(-30.0, -30.0), (30.0, -30.0), (30.0, 30.0), (-30.0, 30.0)]
    bands = shadow.shadow_bands(0, 0, 6, 6, 0, H=4.3, deemed_boundary=boundary, mh=4.0, dt_minutes=15)
    assert bands["max_10plus"] == 0.0  # 影は境界(30m)を越えて10m超帯まで届かない


def test_bands_from_raster_mask_restricts_to_polygon():
    # 「影が落ちる先の区域」（法56条の2第4項）はマスク（対象区域の
    # ポリゴン）の中にしか適用されない、という不具合修正の核心を検証する。
    # 冬至日は太陽が南寄りにあるため、影は北側(+y)に伸びる。南側だけを
    # マスクすると、影が実際に落ちていない場所しか見ないので0になる。
    boundary = [(-30.0, -30.0), (30.0, -30.0), (30.0, 30.0), (-30.0, 30.0)]
    raster = shadow.shadow_raster(0, 0, 10, 10, 0, H=20.0, deemed_boundary=boundary, mh=4.0, dt_minutes=15)
    assert raster is not None
    full = shadow.bands_from_raster(raster)
    assert full["max_10plus"] > 0.0

    south_mask = [(-100.0, -100.0), (100.0, -100.0), (100.0, 0.0), (-100.0, 0.0)]
    south_only = shadow.bands_from_raster(raster, mask_polygon=south_mask)
    assert south_only["max_5_10"] == 0.0
    assert south_only["max_10plus"] == 0.0

    north_mask = [(-100.0, 0.0), (100.0, 0.0), (100.0, 100.0), (-100.0, 100.0)]
    north_only = shadow.bands_from_raster(raster, mask_polygon=north_mask)
    assert math.isclose(north_only["max_10plus"], full["max_10plus"])


def test_shadow_raster_none_when_building_below_measurement_height():
    boundary = [(-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0)]
    assert shadow.shadow_raster(0, 0, 20, 20, 0, H=3.0, deemed_boundary=boundary, mh=4.0) is None
    assert shadow.bands_from_raster(None) == {"max_5_10": 0.0, "max_10plus": 0.0}


def test_shadow_reach_distance_matches_design_formula():
    # 設計書3.5-1: 有効高さ×6.71 (北緯35度、冬至日8時・16時付近が最大倍率)
    d = shadow.shadow_reach_distance(10.0, phi_deg=35.0)
    assert math.isclose(d, 10.0 * 6.71, abs_tol=0.05)


def test_shadow_reach_distance_zero_for_nonpositive_height():
    assert shadow.shadow_reach_distance(0.0) == 0.0
    assert shadow.shadow_reach_distance(-5.0) == 0.0


def test_shadow_bands_matches_hikage_relax_deemed_boundary_shift():
    # 建物が境界のすぐ外側まで届く高さなら、緩和(hikage_relax)で
    # みなし境界線を外側に押し出すほど、同じ実測距離での日影時間が
    # 短く（あるいは同等に）評価されるはず。
    site = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]
    edges_no_relax = [{"type": "rinchi", "width": 0}] * 4
    edges_road = [{"type": "road", "width": 12.0}] * 4  # width>10 -> relax = width-5 = 7

    offsets_no = [reg.hikage_relax(e["type"], e["width"]) for e in edges_no_relax]
    offsets_road = [reg.hikage_relax(e["type"], e["width"]) for e in edges_road]
    assert offsets_no == [0.0, 0.0, 0.0, 0.0]
    assert offsets_road == [7.0, 7.0, 7.0, 7.0]

    dp_no = geo.offset_polygon_per_edge(site, offsets_no)
    dp_road = geo.offset_polygon_per_edge(site, offsets_road)
    assert geo.polygon_area(dp_road) > geo.polygon_area(dp_no)
