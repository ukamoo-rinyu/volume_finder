import math
import os
import sys

# insert the repo root (parent of volume_finder/) so imports are
# package-qualified (volume_finder.core...) and never shadow stdlib `io`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from volume_finder.core import geometry as geo


def test_signed_area_ccw_positive():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert geo.signed_area(square) == 100.0


def test_signed_area_cw_negative():
    square_cw = [(0, 0), (0, 10), (10, 10), (10, 0)]
    assert geo.signed_area(square_cw) == -100.0


def test_to_ccw_full_reverse_on_flip():
    square_cw = [(0, 0), (0, 10), (10, 10), (10, 0)]
    out, flipped = geo.to_ccw(square_cw)
    assert flipped is True
    assert geo.signed_area(out) > 0
    # matches HTML's P.slice().reverse(): the whole list reverses,
    # so index 0 changes too (old last vertex becomes new first)
    assert out == square_cw[::-1]
    assert out[0] == square_cw[-1]


def test_to_ccw_noop_when_already_ccw():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    out, flipped = geo.to_ccw(square)
    assert flipped is False
    assert out == square


def test_polygon_area_matches_known_square():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert geo.polygon_area(square) == 100.0


def test_polygon_centroid_of_square():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    cx, cy = geo.polygon_centroid(square)
    assert math.isclose(cx, 5.0)
    assert math.isclose(cy, 5.0)


def test_simplify_polygon_keeps_start_vertex():
    # a near-straight edge with an extra near-collinear vertex that
    # should be dropped at a coarse tolerance
    poly = [(0, 0), (5, 0.01), (10, 0), (10, 10), (0, 10)]
    simplified = geo.simplify_polygon(poly, tolerance=0.1)
    assert simplified[0] == poly[0]
    assert len(simplified) == 4


def test_simplify_polygon_falls_back_when_too_aggressive():
    triangle = [(0, 0), (10, 0), (5, 10)]
    simplified = geo.simplify_polygon(triangle, tolerance=1000.0)
    assert len(simplified) >= 3


def test_area_change_ratio():
    before = [(0, 0), (10, 0), (10, 10), (0, 10)]
    after = [(0, 0), (10, 0), (10, 10.5), (0, 10.5)]
    ratio = geo.area_change_ratio(before, after)
    assert math.isclose(ratio, 0.05)


def test_largest_ring_picks_max_area_and_counts_discards():
    small = [(0, 0), (1, 0), (1, 1), (0, 1)]
    big = [(0, 0), (100, 0), (100, 100), (0, 100)]
    chosen, idx, discarded = geo.largest_ring([small, big, small])
    assert chosen == big
    assert idx == 1
    assert discarded == 2


def test_polygon_edges_numbering_and_length():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    edges = geo.polygon_edges(square)
    assert [e["index"] for e in edges] == [1, 2, 3, 4]
    assert math.isclose(edges[0]["length"], 10.0)
    # bearing = which way the edge *faces* (outward normal), not the
    # direction you'd walk along it (設計書2.1.2の外向き法線 n=(dy,-dx)).
    # edge 1: (0,0)->(10,0), the bottom edge of a CCW square -> faces south
    assert math.isclose(edges[0]["bearing"], 180.0)
    # edge 2: (10,0)->(10,10), the right edge of a CCW square -> faces east
    assert math.isclose(edges[1]["bearing"], 90.0)
    # edge 3: (10,10)->(0,10), the top edge -> faces north
    assert math.isclose(edges[2]["bearing"], 0.0)
    # edge 4: (0,10)->(0,0), the left edge -> faces west
    assert math.isclose(edges[3]["bearing"], 270.0)


def test_latlng_local_roundtrip():
    origin_lat, origin_lng = 34.6937, 135.5023
    lat, lng = 34.6950, 135.5040
    x, y = geo.latlng_to_local(lat, lng, origin_lat, origin_lng)
    lat2, lng2 = geo.local_to_latlng(x, y, origin_lat, origin_lng)
    assert math.isclose(lat, lat2, abs_tol=1e-9)
    assert math.isclose(lng, lng2, abs_tol=1e-9)


def test_latlng_to_local_origin_is_zero():
    origin_lat, origin_lng = 34.6937, 135.5023
    x, y = geo.latlng_to_local(origin_lat, origin_lng, origin_lat, origin_lng)
    assert math.isclose(x, 0.0, abs_tol=1e-9)
    assert math.isclose(y, 0.0, abs_tol=1e-9)


# --- 段階3で追加したジオメトリ (offset/inscribed rectangle/raster helpers) ---


def test_offset_polygon_per_edge_uniform_inward():
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    shrunk = geo.offset_polygon_per_edge(square, [-5, -5, -5, -5])
    assert shrunk == [(5.0, 5.0), (95.0, 5.0), (95.0, 95.0), (5.0, 95.0)]
    assert math.isclose(geo.polygon_area(shrunk), 90 * 90)


def test_offset_polygon_per_edge_outward():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    grown = geo.offset_polygon_per_edge(square, [2, 2, 2, 2])
    assert grown == [(-2.0, -2.0), (12.0, -2.0), (12.0, 12.0), (-2.0, 12.0)]


def test_offset_polygon_per_edge_known_limitation_on_narrow_notch():
    """既知の限界: 細い切れ込みのある凹んだ敷地では、ミター結合による
    単純なオフセットが指定距離より内側に食い込んだ（＝より安全な）点を
    作ることがあり、その結果できる多角形が自己交差して、指定した距離
    より敷地境界に近い領域まで「内部」と誤判定されることがある。

    これが実際に起きることを示すための特性化テスト（このオフセット
    関数自体を直そうとしているわけではない）。core.search はこの限界を
    知った上で、探索実行時は呼び出し側（ui/dock.py）がQGIS/GEOSの
    buffer()で計算した建築可能範囲を SearchParams.buildable_override
    として渡すことでこれを回避する。単体テストでは軸並行の矩形など
    素直な凸形状しか使っていなかったため、この問題は見つかっていなかった。
    """
    # 上端から細い切れ込みが下に伸びる敷地（切れ込みの先端が凹頂点）。
    site = [
        (0.0, 0.0), (70.0, 0.0), (70.0, 50.0),
        (35.0, 50.0), (33.0, 20.0), (30.0, 50.0), (0.0, 50.0),
    ]
    offset = geo.offset_polygon_per_edge(site, [-1.0] * len(site))
    # 切れ込みの先端付近の頂点は、1mよりはるかに敷地境界から離れた
    # 場所に飛んでしまう（正しいオフセットなら1m前後のはず）。
    tip_offset_point = offset[4]
    distance_from_original_boundary = geo.point_to_polygon_distance(
        np.array([tip_offset_point[0]]), np.array([tip_offset_point[1]]), site
    )[0]
    assert distance_from_original_boundary > 3.0  # 意図した1mを大幅に超える


def test_rectangle_corners_axis_aligned():
    corners = geo.rectangle_corners(0, 0, 10, 4, 0)
    assert set(corners) == {(-5.0, -2.0), (5.0, -2.0), (5.0, 2.0), (-5.0, 2.0)}


def test_point_in_polygon_matches_vectorized_mask():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert geo.point_in_polygon((5, 5), square) is True
    assert geo.point_in_polygon((15, 5), square) is False
    import numpy as np

    X, Y = np.meshgrid(np.array([5.0, 15.0]), np.array([5.0]))
    mask = geo.point_in_polygon_mask(X, Y, square)
    assert mask.tolist() == [[True, False]]


def test_point_to_polygon_distance():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    import numpy as np

    X = np.array([15.0, 5.0])
    Y = np.array([5.0, 5.0])
    d = geo.point_to_polygon_distance(X, Y, square)
    assert math.isclose(d[0], 5.0)
    assert math.isclose(d[1], 5.0)  # inside: distance to nearest edge (5m to any side)


def test_rectangle_in_polygon_true_and_false():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert geo.rectangle_in_polygon((5, 5), (1, 0), (0, 1), 4, 4, square) is True
    assert geo.rectangle_in_polygon((5, 5), (1, 0), (0, 1), 6, 4, square) is False


def test_rectangle_in_polygon_rejects_concave_notch_even_with_corners_inside():
    # C字型: 右側中央がへこんでいるので、4隅が内部でも中心を跨ぐ矩形は入らない
    notched = [(0, 0), (10, 0), (10, 4), (5, 4), (5, 6), (10, 6), (10, 10), (0, 10)]
    assert geo.rectangle_in_polygon((2, 5), (1, 0), (0, 1), 1.5, 4, notched) is True
    assert geo.rectangle_in_polygon((5, 5), (1, 0), (0, 1), 4.5, 4, notched) is False


def test_interior_seed_point_for_convex_polygon_is_centroid():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    seed = geo.interior_seed_point(square)
    assert seed == (5.0, 5.0)


def test_candidate_seed_points_finds_both_arms_of_l_shape():
    lshape = [(0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100)]
    seeds = geo.candidate_seed_points(lshape, k=4)
    assert len(seeds) >= 2
    for s in seeds:
        assert geo.point_in_polygon(s, lshape)


def test_rectangle_for_aspect_ratio_square_fills_square_envelope():
    square = [(0.0, 0.0), (90.0, 0.0), (90.0, 90.0), (0.0, 90.0)]
    rect = geo.rectangle_for_aspect_ratio(square, 0, 1.0, seed=(45, 45))
    assert rect is not None
    assert math.isclose(rect["W"], 90, abs_tol=0.1)
    assert math.isclose(rect["D"], 90, abs_tol=0.1)


def test_rectangle_for_aspect_ratio_respects_ratio():
    square = [(0.0, 0.0), (90.0, 0.0), (90.0, 90.0), (0.0, 90.0)]
    rect = geo.rectangle_for_aspect_ratio(square, 0, 3.0, seed=(45, 45))
    assert rect is not None
    assert math.isclose(rect["W"] / rect["D"], 3.0, rel_tol=0.02)
    assert math.isclose(rect["D"], 30, abs_tol=0.5)


def test_rectangle_for_aspect_ratio_45deg_inscribed_square():
    # 一辺100の正方形に45度回転して内接する最大正方形の一辺は 100/sqrt(2)
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    rect = geo.rectangle_for_aspect_ratio(square, 45, 1.0, seed=(50, 50))
    assert rect is not None
    assert math.isclose(rect["W"], 100 / math.sqrt(2), rel_tol=0.01)


def test_rectangle_for_aspect_ratio_none_when_seed_outside():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    rect = geo.rectangle_for_aspect_ratio(square, 0, 1.0, seed=(50, 50))
    assert rect is None


def test_rectangle_for_aspect_ratio_matches_rectangle_corners_handedness():
    """回帰テスト: rectangle_for_aspect_ratio の内部で使う軸(u,v)は、
    rectangle_corners()（HTMLのcorners(b)を移植したもの、W軸=(cos,-sin)・
    D軸=(sin,cos)という時計回り正の規則）と必ず同じ向きでなければならない。

    向きが食い違うと、「fitする」と判定された(cx,cy,W,D,rot)の組を、
    実際の建物ジオメトリを作るときに使う rectangle_corners() に渡すと、
    回転の向きだけ鏡映された（rot⇔-rot）別の矩形になってしまい、
    fit判定は通ったのに実際には敷地からの離隔が確保できていない、
    という食い違いが起きる（0度・180度回転では対称なため気づきにくい。
    非対称な多角形と90度以外の回転角でないと再現しない）。
    """
    # 非対称な多角形（左右対称でも上下対称でもないL字）。
    lshape = [(0, 0), (40, 0), (40, 20), (20, 20), (20, 40), (0, 40)]
    for rotation_deg in (17, 33, 61, 74):
        for ratio in (0.6, 1.0, 1.8):
            rect = geo.rectangle_for_aspect_ratio(lshape, rotation_deg, ratio)
            if rect is None:
                continue
            corners = geo.rectangle_corners(rect["cx"], rect["cy"], rect["W"], rect["D"], rect["rot"])
            for corner in corners:
                assert geo.point_in_polygon(corner, lshape), (
                    f"rot={rotation_deg} ratio={ratio}: corner {corner} computed via "
                    "rectangle_corners() falls outside the polygon rectangle_for_aspect_ratio "
                    "validated as fitting -- handedness mismatch"
                )
