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
    size_fractions=(1.0, 0.85, 0.7),
    shadow_dt_minutes=15,
    max_results=3,
    pool_size=40,  # 有効空地ラスタ計算の対象を絞ってテストを高速化する
)

# 1候補/回転だけに絞りたいテスト用: サイズを敷地いっぱい(1.0)に固定すると、
# 全アンカーが同じ矩形に収束する（自由軸の可動域が0になるため）。
SINGLE_CANDIDATE_PARAMS = dict(size_fractions=(1.0,))


def _inside_or_on_boundary(point, poly, tol=1e-2):
    """point_in_polygon はレイキャスティングの境界条件依存で、ちょうど
    境界上の点を内外どちらとも判定しうる。アンカー配置は建物の辺が
    建築可能領域の境界に乗ることが多い（それがアンカーの目的）ため、
    テストの包含チェックは境界からtol以内も許容する。"""
    import numpy as np

    from volume_finder.core import geometry as geo_module

    if geo_module.point_in_polygon(point, poly):
        return True
    d = geo_module.point_to_polygon_distance(np.array([point[0]]), np.array([point[1]]), poly)[0]
    return d <= tol


def test_rotation_candidates_edge_mode_is_axis_directions_only():
    edges = [{"type": "rinchi", "width": 0}] * 4
    angles = search._rotation_candidates(SITE, edges, mode="edge")
    assert angles == [0.0]  # 軸並行の矩形敷地なので0度のみ


def test_rotation_candidates_orthogonal_mode_uses_widest_road_edge():
    site = [(0.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)]
    edges = [
        {"type": "road", "width": 6.0},
        {"type": "road", "width": 12.0},  # 辺2(東側、鉛直=90度方向)の方が幅員が広い
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
    ]
    angles = search._rotation_candidates(site, edges, mode="orthogonal")
    assert angles == [0.0]  # 辺2は90度方向 -> mod 90 = 0


def test_rotation_candidates_free_mode_includes_edge_directions_and_step():
    edges = [{"type": "rinchi", "width": 0}] * 4
    angles = search._rotation_candidates(SITE, edges, mode="free", step_deg=30)
    assert 0.0 in angles
    assert all(0 <= a < 90 for a in angles)
    assert 30.0 in angles


def test_dedupe_angles_merges_within_one_degree():
    out = search._dedupe_angles_mod90([10.0, 10.4, 45.0, 89.6])
    # 10.0/10.4は1度以内なのでまとめられる
    assert len(out) == 3


def test_anchor_positions_corner_anchor_is_single_point():
    positions = search._anchor_positions(0, 60, 0, 40, 20, 10, "NE", 1.0, 25)
    assert positions == [(40.0, 30.0)]


def test_anchor_positions_free_axis_slides_across_range():
    positions = search._anchor_positions(0, 60, 0, 40, 20, 10, "N", 1.0, 25)
    # u0は[0, 40]の範囲を1m刻みでスライド、v0は常にVmax-D=30
    assert all(v0 == 30.0 for _u0, v0 in positions)
    us = sorted(u0 for u0, _v0 in positions)
    assert us[0] == 0.0
    assert us[-1] == 40.0


def test_anchor_positions_rejects_oversized_footprint():
    assert search._anchor_positions(0, 60, 0, 40, 999, 10, "C", 1.0, 25) == []


def test_footprint_candidates_stay_within_buildable_envelope():
    buildable = geo.offset_polygon_per_edge(SITE, [-1.0] * 4)
    params = search.SearchParams(size_fractions=(1.0, 0.7))
    rotations = search._rotation_candidates(SITE, [{"type": "rinchi", "width": 0}] * 4, params.rotation_mode)
    footprints = search._footprint_candidates(buildable, params, rotations)
    assert footprints
    for fp in footprints:
        for corner in geo.rectangle_corners(fp["cx"], fp["cy"], fp["W"], fp["D"], fp["rot"]):
            assert _inside_or_on_boundary(corner, buildable)


def test_footprint_candidates_full_size_collapses_all_anchors_to_one_rect():
    buildable = geo.offset_polygon_per_edge(SITE, [-1.0] * 4)
    params = search.SearchParams(**SINGLE_CANDIDATE_PARAMS)
    rotations = search._rotation_candidates(SITE, [{"type": "rinchi", "width": 0}] * 4, params.rotation_mode)
    footprints = search._footprint_candidates(buildable, params, rotations)
    assert len(footprints) == 1  # 全アンカーが同一矩形に重複除去される


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


def test_slant_max_height_concave_site_does_not_use_extended_line_of_distant_corner():
    # 凹んだ敷地（L字）で実際に見つかった不具合の再現。辺3(index2、(30,10)->
    # (10,10))の境界線を延長した直線が、その辺とは無関係な上の腕にある
    # 建物の角をかすめ、後退距離が誤ってほぼ0（許容高さ20m=H0そのもの）に
    # なっていた。修正後は、区間外に射影される角には辺の端点までの実距離
    # を使うため、辺3は効かず、実際に近い辺（辺4/5/6）が効いて25mになる。
    site = [(0.0, 0.0), (30.0, 0.0), (30.0, 10.0), (10.0, 10.0), (10.0, 30.0), (0.0, 30.0)]
    edges = [{"type": "rinchi", "width": 0}] * 6
    corners = geo.rectangle_corners(5.0, 25.0, 6.0, 6.0, 0.0)  # L字の上の腕、辺3の延長線をかすめる位置
    h, binding = search._slant_max_height(corners, site, edges, "1住", 200)
    assert math.isclose(h, 25.0, abs_tol=1e-6)
    assert "隣地斜線(辺3)" not in binding


def test_floors_from_max_height_basic_and_infinite():
    assert search._floors_from_max_height(14.03, 3.5, 20) == 4
    assert search._floors_from_max_height(float("inf"), 3.5, 20) == 20


def test_search_slant_safety_margin_avoids_boundary_exact_floor_count():
    # 許容高さがちょうど階の境界のわずか上（3cm）にある場合、安全余裕
    # (既定5cm)を引くと1階減る。JSON書き出し→HTMLツールでの再計算での
    # 数mmのずれで「超過」に反転しないようにするための余裕（実際に見つかった
    # 不具合の再現）。
    site = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
    edges = [{"type": "rinchi", "width": 0}] * 4
    # 隣地斜線: H0=20, k=1.25 (1住)。後退距離Aを、許容高さがちょうど
    # 4階(floor_height=3.5m→14.0m)の3cm上になるよう逆算する。
    # allow = H0 + k*2*A = 20 + 2.5*A -> 欲しいallowは 4*3.5+0.03=14.03 だが
    # H0=20 > 14.03 なので、代わりに8階(28.0m)の3cm上を使う。
    target_allow = 8 * 3.5 + 0.03  # 28.03
    A = (target_allow - 20) / 2.5
    # 幅10m（左右の後退は5mで固定、allow=20+2.5*5=32.5 > target_allowなので
    # 効かない）、奥行だけ調整して上下の後退をAにする（上下の辺が effective）。
    corners = geo.rectangle_corners(10, 10, 10, 20 - 2 * A, 0)
    slant_h, _binding = search._slant_max_height(corners, site, edges, "1住", 200)
    assert math.isclose(slant_h, target_allow, abs_tol=1e-6)

    margin = search.SearchParams().slant_safety_margin_m
    floors_no_margin = search._floors_from_max_height(slant_h, 3.5, 20)
    floors_with_margin = search._floors_from_max_height(slant_h - margin, 3.5, 20)
    assert floors_no_margin == 8
    assert floors_with_margin == 7


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
    # 上位ほど総合スコアが大きい（降順、仕様3.7の重み付きランキング）
    scores = [c.score for c in results]
    assert scores == sorted(scores, reverse=True)


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
            assert _inside_or_on_boundary(corner, envelope)


def test_search_unregulated_zone_skips_shadow_binding():
    # 商業地域は大阪市日影規制の対象外 (core.regulation.hikage_rule が None)
    edges = [
        {"type": "road", "width": 16.0},  # >=12m: 前面道路幅員による容積率制限なし
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
    ]
    params = search.SearchParams(
        min_setback=1.0, floor_height=3.5, max_floors=25,
        shadow_dt_minutes=20, max_results=2, **SINGLE_CANDIDATE_PARAMS,
    )
    results = search.search(SITE, edges, "商業", 400, 100, params=params)
    assert results
    for c in results:
        assert "日影規制" not in c.binding


# --- 影が落ちる先の用途地域 ---


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
        min_setback=1.0, floor_height=3.5, max_floors=25,
        shadow_dt_minutes=20, max_results=1, **SINGLE_CANDIDATE_PARAMS,
    )
    baseline = search.search(SITE, edges, "商業", 400, 100, params=unregulated_params)
    assert baseline
    assert "日影規制" not in baseline[0].binding

    strict_rule = reg.hikage_rule("1中高", 200)  # t2=2.5h, 大阪市6区分の中で最も厳しい
    override_params = search.SearchParams(
        min_setback=1.0, floor_height=3.5, max_floors=25,
        shadow_dt_minutes=20, max_results=1, **SINGLE_CANDIDATE_PARAMS,
        hikage_regions=[{"mask": None, "rule": strict_rule, "name": "1中高"}],
    )
    overridden = search.search(SITE, edges, "商業", 400, 100, params=override_params)
    assert overridden
    assert overridden[0].floors <= baseline[0].floors
    assert "日影規制（影が落ちる先：1中高）" in overridden[0].binding


def test_floors_within_shadow_masks_target_region_to_its_polygon():
    # 「影が落ちる先の区域」は、そのマスク（対象区域のポリゴン）の中でしか
    # 判定してはいけない（ver0.2.0で修正した不具合）。影が実際には届かない
    # 方角にしかない対象区域は、階数を制限してはならない。
    site = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
    footprint = {"cx": 15.0, "cy": 15.0, "W": 10.0, "D": 10.0, "rot": 0.0}
    strict_rule = {"mh": 4.0, "t1": 0.01, "t2": 0.01}  # ほぼ一切の日影を許さない極端な基準

    # 影が実際に落ちる側(冬至日は北側、+y)に対象区域を置く -> 効くはず
    reachable_mask = [(-100.0, 15.0), (100.0, 15.0), (100.0, 200.0), (-100.0, 200.0)]
    floors_reachable, binding_reachable = search._floors_within_shadow(
        footprint, "商業", 400, site, floor_height=3.5, upper_bound=20,
        dt_minutes=20, cell=None,
        hikage_regions=[{"mask": reachable_mask, "rule": strict_rule, "name": "北側の対象区域"}],
    )
    assert binding_reachable  # 制限された

    # 影が届かない側(南側、-y)にしか対象区域が無ければ -> 効かないはず
    unreachable_mask = [(-100.0, -200.0), (100.0, -200.0), (100.0, -15.0), (-100.0, -15.0)]
    floors_unreachable, binding_unreachable = search._floors_within_shadow(
        footprint, "商業", 400, site, floor_height=3.5, upper_bound=20,
        dt_minutes=20, cell=None,
        hikage_regions=[{"mask": unreachable_mask, "rule": strict_rule, "name": "南側の対象区域"}],
    )
    assert not binding_unreachable
    assert floors_unreachable == 20  # 制限されず上限まで許容される
    assert floors_unreachable > floors_reachable


def test_floors_within_shadow_shares_raster_across_regions_with_same_mh(monkeypatch):
    # 日影ラスタ（時刻ループを含む重い計算）は測定面高さ(mh)ごとに1回だけ
    # 計算し、同じmhの区域どうしで使い回すはず（区域数ぶん繰り返さない）。
    site = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
    footprint = {"cx": 15.0, "cy": 15.0, "W": 10.0, "D": 10.0, "rot": 0.0}
    calls = []
    original = search.sh.shadow_raster

    def spy(*args, **kwargs):
        calls.append(kwargs.get("mh"))
        return original(*args, **kwargs)

    monkeypatch.setattr(search.sh, "shadow_raster", spy)
    regions = [
        {"mask": None, "rule": {"mh": 4.0, "t1": 5, "t2": 3}, "name": "a"},
        {"mask": None, "rule": {"mh": 4.0, "t1": 3, "t2": 2}, "name": "b"},  # aと同じmh
        {"mask": None, "rule": {"mh": 6.5, "t1": 5, "t2": 3}, "name": "c"},  # 別のmh
    ]
    search._floors_within_shadow(
        footprint, "商業", 400, site, floor_height=3.5, upper_bound=3,
        dt_minutes=20, cell=None, hikage_regions=regions,
    )
    assert len(calls) == 2  # 3区域でも、mhの種類(4.0と6.5)ぶんの2回だけ
    assert set(calls) == {4.0, 6.5}


def test_search_respects_max_floors_cap():
    edges = [
        {"type": "road", "width": 20.0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
    ]
    params = search.SearchParams(
        min_setback=1.0, floor_height=3.0, max_floors=2,
        shadow_dt_minutes=20, max_results=1, **SINGLE_CANDIDATE_PARAMS,
    )
    results = search.search(SITE, edges, "商業", 1300, 100, params=params)
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


def test_search_pool_out_lets_caller_rerank_without_research():
    edges = [
        {"type": "road", "width": 8.0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
    ]
    site_area = geo.polygon_area(SITE)
    pool = []
    default_results = search.search(SITE, edges, "1住", 200, 60, params=FAST_PARAMS, pool_out=pool)
    assert pool  # プールに正規化済みの候補が入っている
    assert all(c.norm for c in pool)

    # 容積のみを重視する重みでは、floor_areaが最大の候補がトップに来るはず
    far_only = search.rerank(pool, {"far": 1.0, "open": 0, "light": 0, "impact": 0}, site_area, max_results=3)
    assert far_only[0].floor_area == max(c.floor_area for c in pool)

    # 元のsearch()結果と同じ重みで作り直せば、再ランキング後も同じ1位になる
    weights = {"far": FAST_PARAMS.weight_far, "open": FAST_PARAMS.weight_open,
               "light": FAST_PARAMS.weight_light, "impact": FAST_PARAMS.weight_impact}
    rebuilt = search.rerank(pool, weights, site_area, FAST_PARAMS.dedupe_threshold, FAST_PARAMS.max_results)
    assert rebuilt[0].floor_area == default_results[0].floor_area


# --- アンカーごとの比較 (ver0.3.0) ---


def test_top_candidates_by_anchor_differ_in_position_for_partial_size():
    edges = [
        {"type": "road", "width": 8.0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
        {"type": "rinchi", "width": 0},
    ]
    pool = []
    search.search(SITE, edges, "1住", 200, 60, params=FAST_PARAMS, pool_out=pool)
    weights = {"far": FAST_PARAMS.weight_far, "open": FAST_PARAMS.weight_open,
               "light": FAST_PARAMS.weight_light, "impact": FAST_PARAMS.weight_impact}
    by_anchor = search.top_candidates_by_anchor(pool, weights, anchors=("N", "S"))
    assert set(by_anchor.keys()) == {"N", "S"}
    assert by_anchor["N"] is not None
    assert by_anchor["S"] is not None
    # 北寄せ(N)は南寄せ(S)よりcy(=世界座標のy、この敷地では北方向)が大きいはず
    assert by_anchor["N"].cy > by_anchor["S"].cy


def test_top_candidates_by_anchor_returns_none_for_missing_anchor():
    by_anchor = search.top_candidates_by_anchor(
        [], {"far": 1.0, "open": 0, "light": 0, "impact": 0}, anchors=("N", "C")
    )
    assert by_anchor == {"N": None, "C": None}


# --- 複数用途地域での探索 ---


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
        min_setback=1.0, floor_height=3.0, max_floors=30,
        shadow_dt_minutes=30, max_results=1, **SINGLE_CANDIDATE_PARAMS,
    )
    # 道路が無いと容積率上限が0%になるため、比較対象は隣地斜線だけにしたいが
    # far制限も外したいので、道路のない状態のままだと全滅してしまう。
    # そこで容積率の制約を実質無視できるよう far を大きくしすぎない代わりに、
    # _slant_max_height 側の比較で十分検証できているので、ここでは
    # 前面道路をつけて容積率制限を外す。
    edges_with_road = [{"type": "road", "width": 20.0}, {"type": "rinchi", "width": 0},
                        {"type": "rinchi", "width": 0}, {"type": "rinchi", "width": 0}]

    baseline = search.search(site, edges_with_road, "1住", 1300, 100, params=base_params)
    assert baseline

    zone_regions = [{"zone_key": "商業", "far": 1300.0, "polygon": list(site)}]
    override_params = search.SearchParams(
        min_setback=1.0, floor_height=3.0, max_floors=30,
        shadow_dt_minutes=30, max_results=1, zone_regions=zone_regions, **SINGLE_CANDIDATE_PARAMS,
    )
    overridden = search.search(site, edges_with_road, "1住", 1300, 100, params=override_params)
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
        max_floors=20, shadow_dt_minutes=30, max_results=1, **SINGLE_CANDIDATE_PARAMS,
    )
    results = search.search(SITE, edges, "1住", 200, 60, params=params)
    assert results
    for corner in geo.rectangle_corners(results[0].cx, results[0].cy, results[0].W, results[0].D, results[0].rot):
        assert _inside_or_on_boundary(corner, tiny_buildable)


def test_search_buildable_override_empty_yields_no_results():
    params = search.SearchParams(buildable_override=[], max_results=1)
    edges = [{"type": "road", "width": 8.0}] + [{"type": "rinchi", "width": 0}] * 3
    assert search.search(SITE, edges, "1住", 200, 60, params=params) == []


# --- 複数棟配置 (ver0.3.0、分棟・L型) ---


def test_zone_bounds_splits_and_applies_gap_only_to_second_zone():
    bounds_a, bounds_b = search._zone_bounds("u", 0.5, 3.0, 0.0, 60.0, 0.0, 40.0)
    assert bounds_a == (0.0, 30.0, 0.0, 40.0)
    assert bounds_b == (33.0, 60.0, 0.0, 40.0)  # 境界+gapから始まる

    bounds_a_v, bounds_b_v = search._zone_bounds("v", 0.25, 0.0, 0.0, 60.0, 0.0, 40.0)
    assert bounds_a_v == (0.0, 60.0, 0.0, 10.0)
    assert bounds_b_v == (0.0, 60.0, 10.0, 40.0)  # gap=0ならすき間なく接する(L型)


def test_footprint_candidates_local_bounds_override_restricts_to_zone():
    buildable = geo.offset_polygon_per_edge(SITE, [-1.0] * 4)
    params = search.SearchParams(size_fractions=(1.0,))
    zone_bounds = (1.0, 30.0, 1.0, 39.0)  # 敷地の西半分だけ
    fps = search._footprint_candidates(buildable, params, [0.0], local_bounds_override=zone_bounds)
    assert fps
    for fp in fps:
        assert fp["cx"] <= 30.0 + 1e-6  # ゾーンの外(東側)には出ない


def test_top_k_by_distinct_size_prefers_size_diversity_over_duplicates():
    # 同じW×Dでアンカー(スライド位置)違いの候補が大量にあっても、
    # 上位k件はサイズの異なる候補で埋まるべき（同一サイズの重複ではなく）。
    fps = [{"W": 10.0, "D": 10.0, "anchor": f"N{i}"} for i in range(20)]  # 大きいが同一サイズ
    fps += [{"W": 5.0, "D": 5.0, "anchor": "C"}]  # 小さいが唯一のサイズ
    top = search._top_k_by_distinct_size(fps, 3)
    sizes = {(fp["W"], fp["D"]) for fp in top}
    assert (5.0, 5.0) in sizes  # 小さい方も必ず含まれる
    assert len(top) == 2  # 実際に異なるサイズは2種類しかない


def test_slant_max_height_uses_shared_minimum_setback_across_buildings():
    # 法56条2項の後退距離緩和は、同一敷地に複数棟あるとき棟ごとではなく
    # 敷地内最小の後退距離を全棟共通で使う（このセッションでHTML側にも
    # 入れた修正と同じ規制の性質）。corners_a+corners_bのように2棟分の
    # 角点をまとめて渡すだけで、既存の_slant_max_height がそのまま
    # この規則どおりに動くことを確認する。
    site = [(0.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)]
    edges = [{"type": "road", "width": 8.0}] + [{"type": "rinchi", "width": 0}] * 3
    corners_a = geo.rectangle_corners(30.0, 25.0, 10.0, 10.0, 0.0)  # 道路(y=0)から20m後退
    corners_b = geo.rectangle_corners(30.0, 5.0, 10.0, 10.0, 0.0)  # 道路にほぼ接する(後退0m)

    allow_a_alone, _ = search._slant_max_height(corners_a, site, edges, "1住", 200)
    allow_combined, _binding = search._slant_max_height(corners_a + corners_b, site, edges, "1住", 200)

    R = reg.road_rule("1住", 200)
    assert math.isclose(allow_combined, R["k"] * edges[0]["width"], abs_tol=1e-6)  # A=0mでの道路斜線
    assert allow_combined < allow_a_alone  # B棟のせいでA棟単独より厳しくなる


def test_floors_within_shadow_pair_interaction_reduces_below_solo_sum():
    # 2棟が近接すると、単独ならそれぞれ許容される階数の組合せでも、
    # 合成した影は日影規制を満たさないことがある。
    site = [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)]
    strict_rule = {"mh": 4.0, "t1": 50.0, "t2": 0.3}
    regions = [{"mask": None, "rule": strict_rule, "name": "t"}]
    footprint_a = {"cx": 10.0, "cy": 10.0, "W": 6.0, "D": 6.0, "rot": 0.0}
    footprint_b = {"cx": 30.0, "cy": 30.0, "W": 6.0, "D": 6.0, "rot": 0.0}

    solo_a, _ = search._floors_within_shadow(footprint_a, "商業", 400, site, 3.5, 6, 20, None, hikage_regions=regions)
    solo_b, _ = search._floors_within_shadow(footprint_b, "商業", 400, site, 3.5, 6, 20, None, hikage_regions=regions)
    assert solo_a > 0 and solo_b > 0

    floors_a, floors_b, binding = search._floors_within_shadow_pair(
        footprint_a, footprint_b, "商業", 400, site, 3.5, 3.5, 6, 6, 20, None, hikage_regions=regions,
    )
    assert binding  # 日影規制で頭打ちになった
    assert floors_a != floors_b  # 棟ごとに独立して決まり、強制的に揃えない
    assert 36 * floors_a + 36 * floors_b < 36 * solo_a + 36 * solo_b  # 単独の合計より合成後の方が厳しい


def test_shadow_raster_extra_buildings_unions_masks_not_sums():
    site = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
    kwargs = dict(cx=15.0, cy=15.0, W=6.0, D=6.0, rot_deg=0.0, H=15.0, deemed_boundary=site, mh=4.0, dt_minutes=20)
    solo = search.sh.shadow_raster(**kwargs)

    # 同じ棟をextra_buildingsにも重複して渡しても、影は同じ場所・同じ
    # 時刻に重なるだけなので、合計時間は単独のときと変わらないはず
    # （単純合算ではなく論理和(OR)で積算しているため）。
    duped = search.sh.shadow_raster(
        **kwargs,
        extra_buildings=[{"cx": kwargs["cx"], "cy": kwargs["cy"], "W": kwargs["W"], "D": kwargs["D"],
                           "rot_deg": kwargs["rot_deg"], "H": kwargs["H"]}],
    )
    assert (solo["total"] == duped["total"]).all()

    # 遠く離れた別棟を足すと、その棟の影を捉えるためにグリッド範囲が
    # 拡張されるはず（extra_buildingsが実際に計算へ反映されている証拠）。
    far_away = search.sh.shadow_raster(
        **kwargs, extra_buildings=[{"cx": -300.0, "cy": -300.0, "W": 6.0, "D": 6.0, "rot_deg": 0.0, "H": 15.0}],
    )
    assert far_away["X"].min() < solo["X"].min() - 100


def test_search_two_buildings_returns_non_overlapping_legal_pair():
    edges = [{"type": "road", "width": 8.0}] + [{"type": "rinchi", "width": 0}] * 3
    params = search.SearchParams(
        min_setback=1.0, floor_height=3.5, max_floors=20,
        size_fractions=(1.0, 0.85, 0.7), shadow_dt_minutes=20, max_results=3, pool_size=20,
        pair_zone_top_k=5, building_gap=3.0, split_fractions=(0.4, 0.5, 0.6),
    )
    results = search.search_two_buildings(SITE, edges, "1住", 200, 60, params=params)
    assert results
    top = results[0]
    assert len(top.buildings) == 2
    a, b = top.buildings
    assert math.isclose(top.footprint_area, a.footprint_area + b.footprint_area)
    assert math.isclose(top.floor_area, a.floor_area + b.floor_area)
    assert top.bcr_pct <= 60.0 + 1e-6
    # rot=0(軸並行)のはずなので、u軸(x)方向に離隔ぶんの隙間があいているか、
    # v軸(y)方向に隙間があいているかのどちらかで、必ず重ならない。
    ax0, ax1 = a.cx - a.W / 2, a.cx + a.W / 2
    bx0, bx1 = b.cx - b.W / 2, b.cx + b.W / 2
    ay0, ay1 = a.cy - a.D / 2, a.cy + a.D / 2
    by0, by1 = b.cy - b.D / 2, b.cy + b.D / 2
    separated_x = ax1 <= bx0 + 1e-6 or bx1 <= ax0 + 1e-6
    separated_y = ay1 <= by0 + 1e-6 or by1 <= ay0 + 1e-6
    assert separated_x or separated_y


def test_search_two_buildings_zero_gap_allows_touching_placement():
    edges = [{"type": "road", "width": 8.0}] + [{"type": "rinchi", "width": 0}] * 3
    params = search.SearchParams(
        min_setback=1.0, floor_height=3.5, max_floors=20,
        size_fractions=(1.0, 0.85, 0.7), shadow_dt_minutes=20, max_results=3, pool_size=20,
        pair_zone_top_k=5, building_gap=0.0, split_fractions=(0.4, 0.5, 0.6),
    )
    results = search.search_two_buildings(SITE, edges, "1住", 200, 60, params=params)
    assert results  # 隙間なし(L型)でも探索でき、結果が出る


# --- 施設タイプ (ver0.3.0、庁舎・スポーツ施設・保育所) ---


def test_passes_shape_constraints_enforces_aspect_ratio_dim_and_area():
    ftype = search.FacilityType(
        key="t", name="t", floor_height=3.5,
        max_aspect_ratio=2.0, min_footprint_dim=10.0, min_footprint_area=150.0,
    )
    assert search._passes_shape_constraints(20.0, 15.0, 300.0, ftype)  # 縦横比1.33・短辺15m・面積300m2 -> OK
    assert not search._passes_shape_constraints(30.0, 10.0, 300.0, ftype)  # 縦横比3.0 > 2.0
    assert not search._passes_shape_constraints(8.0, 8.0, 64.0, ftype)  # 短辺8m<10m・面積64m2<150m2
    assert search._passes_shape_constraints(20.0, 15.0, 300.0, None)  # ftype未指定なら常に通す


def test_zone_polygon_matches_bounds_for_unrotated_zone():
    poly = search._zone_polygon((0.0, 10.0, 0.0, 20.0), 0.0)
    xs = sorted(round(p[0], 6) for p in poly)
    ys = sorted(round(p[1], 6) for p in poly)
    assert xs == [0.0, 0.0, 10.0, 10.0]
    assert ys == [0.0, 0.0, 20.0, 20.0]


def test_yard_ok_requires_min_area_and_side():
    region = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
    corners = geo.rectangle_corners(10.0, 10.0, 15.0, 15.0, 0.0)
    ftype_ok = search.FacilityType(
        key="t", name="t", floor_height=3.0, requires_yard=True, yard_min_area=200.0, yard_min_side=10.0
    )
    assert search._yard_ok(region, corners, ftype_ok, cell=1.0)
    ftype_too_big = search.FacilityType(
        key="t2", name="t2", floor_height=3.0, requires_yard=True, yard_min_area=2000.0, yard_min_side=10.0
    )
    assert not search._yard_ok(region, corners, ftype_too_big, cell=1.0)
    assert search._yard_ok(region, corners, None, cell=1.0)  # requires_yard=False(既定)なら常に通す


def test_search_facility_type_constrains_proportions_and_floor_range():
    site = [(0.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)]
    edges = [{"type": "road", "width": 16.0}] + [{"type": "rinchi", "width": 0}] * 3  # 12m以上->容積率制限なし
    ftype = search.FACILITY_TYPES["sports"]
    params = search.SearchParams(
        min_setback=1.0, max_floors=25, shadow_dt_minutes=30, max_results=3,
        size_fractions=(1.0, 0.85, 0.7), facility_type=ftype,
    )
    results = search.search(site, edges, "商業", 400, 60, params=params)
    assert results
    for c in results:
        short_side, long_side = (c.W, c.D) if c.W <= c.D else (c.D, c.W)
        assert short_side >= ftype.min_footprint_dim - 1e-6
        assert long_side <= short_side * ftype.max_aspect_ratio + 1e-6
        assert c.footprint_area >= ftype.min_footprint_area - 1e-6
        assert ftype.min_floors <= c.floors <= ftype.max_floors
        assert math.isclose(c.height, c.floors * ftype.floor_height)


def test_search_two_buildings_building_types_apply_independently_per_zone():
    site = [(0.0, 0.0), (80.0, 0.0), (80.0, 50.0), (0.0, 50.0)]
    edges = [{"type": "road", "width": 16.0}] + [{"type": "rinchi", "width": 0}] * 3
    office = search.FACILITY_TYPES["office"]
    nursery = search.FACILITY_TYPES["nursery"]
    params = search.SearchParams(
        min_setback=1.0, max_floors=25, shadow_dt_minutes=30, max_results=2, pool_size=20,
        size_fractions=(1.0, 0.85, 0.7), pair_zone_top_k=5, building_gap=3.0,
        split_fractions=(0.4, 0.5, 0.6), building_types=(office, nursery),
    )
    results = search.search_two_buildings(site, edges, "商業", 400, 60, params=params)
    assert results
    for c in results:
        a, b = c.buildings
        # A棟(庁舎)・B棟(保育所)で階高が異なることが反映されているはず
        assert math.isclose(a.height, a.floors * office.floor_height)
        assert math.isclose(b.height, b.floors * nursery.floor_height)
        assert office.min_floors <= a.floors <= office.max_floors
        assert nursery.min_floors <= b.floors <= nursery.max_floors
