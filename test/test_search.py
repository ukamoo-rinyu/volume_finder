import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from volume_finder.core import geometry as geo
from volume_finder.core import regulation as reg
from volume_finder.core import search


SITE = [(0.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)]

# 粗いパラメータでテストを高速化する（探索の正しさは検証するが、
# 本番相当の細かい刻みは要らない）。
FAST_PARAMS = search.SearchParams(
    min_setback=1.0,
    floor_height=3.5,
    max_floors=20,
    rotation_step_deg=30,
    aspect_ratios=(0.7, 1.0, 1.5),
    shrink_factors=(1.0, 0.85),
    shadow_dt_minutes=15,
    max_results=3,
)


def test_rotation_candidates_include_edge_directions_and_step():
    angles = search._rotation_candidates(SITE, 30)
    assert 0.0 in angles  # 敷地は軸に平行な矩形なので0度を含む
    assert all(0 <= a < 90 for a in angles)
    assert 30.0 in angles


def test_far_limit_zero_without_road_edges():
    edges = [{"type": "rinchi", "width": 0}] * 4
    lim, wmax = search._far_limit("1住", 200, edges)
    assert lim == 0
    assert wmax == 0


def test_far_limit_capped_by_narrow_road():
    edges = [{"type": "road", "width": 6.0}] * 1 + [{"type": "rinchi", "width": 0}] * 3
    lim, wmax = search._far_limit("1住", 200, edges)
    # farCoef(1住)=0.4, 6m*0.4*100=240% > 200(指定) -> 指定容積率が効く
    assert lim == 200
    edges2 = [{"type": "road", "width": 3.0}] + [{"type": "rinchi", "width": 0}] * 3
    lim2, _ = search._far_limit("1住", 200, edges2)
    assert math.isclose(lim2, 3.0 * 0.4 * 100)


def test_far_limit_unrestricted_when_road_12m_or_more():
    edges = [{"type": "road", "width": 12.0}] + [{"type": "rinchi", "width": 0}] * 3
    lim, wmax = search._far_limit("1住", 200, edges)
    assert lim == 200
    assert wmax == 12.0


def test_bcr_ok_respects_limit_and_relax():
    ok, lim = search._bcr_ok(50, 100, 60, "0")
    assert ok is True and lim == 60
    ok2, lim2 = search._bcr_ok(65, 100, 60, "0")
    assert ok2 is False
    ok3, lim3 = search._bcr_ok(65, 100, 60, "10")
    assert ok3 is True and lim3 == 70


def test_slant_max_height_road_within_applicable_distance():
    site = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
    edges = [{"type": "road", "width": 8.0}, {"type": "rinchi", "width": 0}, {"type": "rinchi", "width": 0}, {"type": "rinchi", "width": 0}]
    corners = geo.rectangle_corners(10, 10, 10, 10, 0)  # building spans y=5..15, so A to edge0(y=0) is 5
    h, binding = search._slant_max_height(corners, site, edges, "1住", 200)
    R = reg.road_rule("1住", 200)
    d_start = 8.0 + 2 * 5.0
    assert d_start <= R["L"]
    assert math.isclose(h, R["k"] * d_start)
    assert binding == ["道路斜線(辺1)"]


def test_slant_max_height_beyond_applicable_distance_is_unrestricted_by_that_edge():
    # 適用距離Lを超える後退があれば、その辺からの制限は外れる
    site = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)]
    edges = [{"type": "road", "width": 8.0}] + [{"type": "rinchi", "width": 0}] * 3
    corners = geo.rectangle_corners(100, 100, 10, 10, 0)  # far from the road edge
    h, binding = search._slant_max_height(corners, site, edges, "1住", 200)
    # 隣地斜線側の制限だけが効くはず（道路斜線の辺はbeyondでスキップされる）
    assert "道路斜線(辺1)" not in binding


def test_slant_max_height_rin_rule_water_gets_half_width_bonus():
    # 4辺とも同条件にして比較する: 1辺だけ変えても、他の辺がまだ効いていて
    # 全体のminが動かないことがあるため（隣地斜線はどの辺も対称に効く）。
    site = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
    edges_rinchi = [{"type": "rinchi", "width": 0}] * 4
    edges_sui = [{"type": "sui", "width": 6.0}] * 4
    corners = geo.rectangle_corners(10, 10, 10, 10, 0)
    h_rinchi, _ = search._slant_max_height(corners, site, edges_rinchi, "1住", 200)
    h_sui, _ = search._slant_max_height(corners, site, edges_sui, "1住", 200)
    assert h_sui > h_rinchi  # 水面の半分ボーナスぶん、許容高さが増える
    assert math.isclose(h_rinchi, 20 + 1.25 * 2 * 5)
    assert math.isclose(h_sui, 20 + 1.25 * 2 * (5 + 3))


def test_search_empty_without_road_frontage():
    edges = [{"type": "rinchi", "width": 0}] * 4
    results = search.search(SITE, edges, "1住", 200, 60, params=FAST_PARAMS)
    assert results == []


def test_search_returns_ranked_candidates_within_far_and_bcr_limits():
    edges = [
        {"type": "road", "width": 8.0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
    ]
    results = search.search(SITE, edges, "1住", 200, 60, params=FAST_PARAMS)
    assert results
    assert [c.rank for c in results] == list(range(1, len(results) + 1))
    for c in results:
        assert c.floors > 0
        assert c.far_pct <= 200 + 0.5
        assert c.bcr_pct <= 60 + 0.5
        assert c.binding
    # 上位ほど延床面積が大きい（降順）
    areas = [c.floor_area for c in results]
    assert areas == sorted(areas, reverse=True)


def test_search_footprints_fit_within_setback_envelope():
    edges = [
        {"type": "road", "width": 8.0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
    ]
    results = search.search(SITE, edges, "1住", 200, 60, params=FAST_PARAMS)
    envelope = geo.offset_polygon_per_edge(SITE, [-FAST_PARAMS.min_setback] * 4)
    for c in results:
        for corner in geo.rectangle_corners(c.cx, c.cy, c.W, c.D, c.rot):
            assert geo.point_in_polygon(corner, envelope)


def test_search_unregulated_zone_skips_shadow_binding():
    # 商業地域は大阪市日影規制の対象外 (core.regulation.hikage_rule が None)
    edges = [
        {"type": "road", "width": 16.0},  # >=12m: 前面道路幅員による容積率制限なし
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
    ]
    params = search.SearchParams(
        min_setback=1.0, floor_height=3.5, max_floors=25, rotation_step_deg=45,
        aspect_ratios=(1.0,), shrink_factors=(1.0,), shadow_dt_minutes=20, max_results=2,
    )
    results = search.search(SITE, edges, "商業", 400, 80, params=params)
    assert results
    for c in results:
        assert "日影規制" not in c.binding


# --- 段階4: 影が落ちる先の用途地域 (設計書3.5、法56条の2第4項) ---


def test_hikage_override_applies_even_when_own_zone_unregulated():
    # 商業地域そのものは日影規制の対象外だが、影が落ちる先（周辺の対象
    # 区域）の基準を hikage_override として渡すと、そちらで制限される。
    edges = [
        {"type": "road", "width": 16.0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
    ]
    unregulated_params = search.SearchParams(
        min_setback=1.0, floor_height=3.5, max_floors=25, rotation_step_deg=45,
        aspect_ratios=(1.0,), shrink_factors=(1.0,), shadow_dt_minutes=20, max_results=1,
    )
    baseline = search.search(SITE, edges, "商業", 400, 80, params=unregulated_params)
    assert baseline
    assert "日影規制" not in baseline[0].binding

    strict_rule = reg.hikage_rule("1中高", 200)  # t2=2.5h, 大阪市6区分の中で最も厳しい
    override_params = search.SearchParams(
        min_setback=1.0, floor_height=3.5, max_floors=25, rotation_step_deg=45,
        aspect_ratios=(1.0,), shrink_factors=(1.0,), shadow_dt_minutes=20, max_results=1,
        hikage_override=strict_rule,
    )
    overridden = search.search(SITE, edges, "商業", 400, 80, params=override_params)
    assert overridden
    assert overridden[0].floors <= baseline[0].floors
    assert "日影規制（影が落ちる先）" in overridden[0].binding


def test_floors_within_shadow_uses_override_mh_not_a_fixed_value():
    site = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
    footprint = {"cx": 15.0, "cy": 15.0, "W": 10.0, "D": 10.0, "rot": 0.0}
    override = {"mh": 6.5, "t1": 5, "t2": 3}  # 準工業のmhは6.5m（他は4m）
    floors, rule = search._floors_within_shadow(
        footprint, "商業", 400, site, floor_height=3.5, upper_bound=3,
        dt_minutes=20, cell=None, hikage_override=override,
    )
    assert rule["mh"] == 6.5


def test_search_respects_max_floors_cap():
    edges = [
        {"type": "road", "width": 20.0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
    ]
    params = search.SearchParams(
        min_setback=1.0, floor_height=3.0, max_floors=2, rotation_step_deg=45,
        aspect_ratios=(1.0,), shrink_factors=(1.0,), shadow_dt_minutes=20, max_results=1,
    )
    results = search.search(SITE, edges, "商業", 1300, 80, params=params)
    assert results
    assert results[0].floors <= 2


def test_search_progress_callback_and_stop():
    edges = [
        {"type": "road", "width": 8.0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
    ]
    calls = []
    results = search.search(
        SITE, edges, "1住", 200, 60, params=FAST_PARAMS,
        progress_callback=lambda done, total, best: calls.append((done, total)),
        should_stop=lambda: len(calls) >= 3,
    )
    assert len(calls) == 3
    assert calls[0][1] == calls[-1][1]  # total は変わらない
    assert isinstance(results, list)


# --- 段階5: 複数用途地域での探索 (設計書3.6) ---


def test_local_zone_picks_containing_region():
    regions = [
        {"zone_key": "1住", "far": 200.0, "polygon": [(0, 0), (30, 0), (30, 40), (0, 40)]},
        {"zone_key": "商業", "far": 400.0, "polygon": [(30, 0), (60, 0), (60, 40), (30, 40)]},
    ]
    assert search._local_zone(10, 20, regions, "1住", 200) == ("1住", 200.0)
    assert search._local_zone(50, 20, regions, "1住", 200) == ("商業", 400.0)


def test_local_zone_falls_back_when_outside_all_regions():
    regions = [{"zone_key": "1住", "far": 200.0, "polygon": [(0, 0), (10, 0), (10, 10), (0, 10)]}]
    assert search._local_zone(500, 500, regions, "商業", 400) == ("商業", 400)


def test_local_zone_falls_back_when_no_regions_given():
    assert search._local_zone(5, 5, None, "商業", 400) == ("商業", 400)
    assert search._local_zone(5, 5, [], "商業", 400) == ("商業", 400)


def test_slant_max_height_differs_by_local_zone():
    # 同じ後退距離でも、区分が違えば隣地斜線の許容高さが変わる
    # (1住: H0=20,k=1.25 / 商業: H0=31,k=2.5)。
    site = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
    edges = [{"type": "rinchi", "width": 0}] * 4
    corners = geo.rectangle_corners(10, 10, 10, 10, 0)
    h_1ju, _ = search._slant_max_height(corners, site, edges, "1住", 200)
    h_shogyo, _ = search._slant_max_height(corners, site, edges, "商業", 400)
    assert h_shogyo > h_1ju


def test_search_uses_zone_regions_override_for_slant_and_binding_label():
    # 敷地全体をカバーする1つの用途地域領域(商業)を渡すと、search()に
    # 渡したグローバルなzone("1住")より緩い隣地斜線基準が使われ、
    # 高さが伸びるはず。すべての候補の重心は必ずこの領域に入るので
    # (敷地全体を覆っているため)、結果は決定的に比較できる。
    edges = [{"type": "rinchi", "width": 0}] * 4  # 隣地斜線だけが効く単純なケース
    site = [(0.0, 0.0), (30.0, 0.0), (30.0, 20.0), (0.0, 20.0)]
    base_params = search.SearchParams(
        min_setback=1.0, floor_height=3.0, max_floors=30, rotation_step_deg=90,
        aspect_ratios=(1.0,), shrink_factors=(1.0,), shadow_dt_minutes=30, max_results=1,
    )
    # 道路が無いと容積率上限が0%になるため、比較対象は隣地斜線だけにしたいが
    # far制限も外したいので、道路のない状態のままだと全滅してしまう。
    # そこで容積率の制約を実質無視できるよう far を大きくしすぎない代わりに、
    # _slant_max_height 側の比較で十分検証できているので、ここでは
    # 前面道路をつけて容積率制限を外す。
    edges_with_road = [{"type": "road", "width": 20.0}, {"type": "rinchi", "width": 0},
                        {"type": "rinchi", "width": 0}, {"type": "rinchi", "width": 0}]

    baseline = search.search(site, edges_with_road, "1住", 1300, 80, params=base_params)
    assert baseline

    zone_regions = [{"zone_key": "商業", "far": 1300.0, "polygon": list(site)}]
    override_params = search.SearchParams(
        min_setback=1.0, floor_height=3.0, max_floors=30, rotation_step_deg=90,
        aspect_ratios=(1.0,), shrink_factors=(1.0,), shadow_dt_minutes=30, max_results=1,
        zone_regions=zone_regions,
    )
    overridden = search.search(site, edges_with_road, "1住", 1300, 80, params=override_params)
    assert overridden
    assert overridden[0].height >= baseline[0].height
    assert any("[商業]" in label for label in overridden[0].binding)


def test_search_zone_regions_matching_global_zone_is_a_no_op():
    edges = [{"type": "road", "width": 8.0}, {"type": "rinchi", "width": 0},
              {"type": "rinchi", "width": 0}, {"type": "rinchi", "width": 0}]
    without = search.search(SITE, edges, "1住", 200, 60, params=FAST_PARAMS)
    zone_regions = [{"zone_key": "1住", "far": 200.0, "polygon": list(SITE)}]
    with_matching = search.search(
        SITE, edges, "1住", 200, 60,
        params=search.SearchParams(**{**FAST_PARAMS.__dict__, "zone_regions": zone_regions}),
    )
    assert [c.floor_area for c in without] == [c.floor_area for c in with_matching]


# --- buildable_override (実行環境側でGEOS bufferを使う回避策の配線確認) ---


def test_search_uses_buildable_override_instead_of_min_setback_offset():
    # buildable_override を、min_setback由来のオフセットとは明らかに
    # 違う（もっと小さい）矩形にすると、その範囲に収まる候補しか
    # 出てこないはず。core.search が override を実際に使っている証拠になる。
    edges = [{"type": "road", "width": 8.0}, {"type": "rinchi", "width": 0},
              {"type": "rinchi", "width": 0}, {"type": "rinchi", "width": 0}]
    tiny_buildable = [(20.0, 10.0), (40.0, 10.0), (40.0, 30.0), (20.0, 30.0)]  # 20x20 のみ
    params = search.SearchParams(
        min_setback=1.0, buildable_override=tiny_buildable, floor_height=3.5,
        max_floors=20, rotation_step_deg=90, aspect_ratios=(1.0,), shrink_factors=(1.0,),
        shadow_dt_minutes=30, max_results=1,
    )
    results = search.search(SITE, edges, "1住", 200, 60, params=params)
    assert results
    for corner in geo.rectangle_corners(results[0].cx, results[0].cy, results[0].W, results[0].D, results[0].rot):
        assert geo.point_in_polygon(corner, tiny_buildable)


def test_search_buildable_override_empty_yields_no_results():
    params = search.SearchParams(buildable_override=[], max_results=1)
    edges = [{"type": "road", "width": 8.0}] + [{"type": "rinchi", "width": 0}] * 3
    assert search.search(SITE, edges, "1住", 200, 60, params=params) == []
