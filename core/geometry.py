"""Pure Python/numpy geometry helpers.

No QGIS import here by design (see 設計書 6章): this module must be
importable and unit-testable without a running QGIS process, and is a
candidate for reuse in a future CLI tool.

Coordinate/ordering conventions (設計書 2.1.2, 2.4) are chosen to match
the existing HTML tool (hikage-osaka-v4_23.html) exactly:
  - polygons are normalized to CCW the same way the HTML tool's ccw()
    does: a full list reversal when the input is CW, so index 0 can
    change on flip (see to_ccw docstring) — callers must track that
  - local XY <-> lat/lng conversion uses the same equirectangular
    approximation as the HTML tool's mPerDeg()/xy()/ll() functions,
    not a generic projected CRS, so JSON round-trips identically.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def signed_area(pts: Sequence[Point]) -> float:
    """Shoelace signed area. Positive when pts is CCW."""
    s = 0.0
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        s += ax * by - bx * ay
    return s / 2.0


def polygon_area(pts: Sequence[Point]) -> float:
    """Unsigned polygon area."""
    return abs(signed_area(pts))


def to_ccw(pts: Sequence[Point]) -> Tuple[List[Point], bool]:
    """Normalize winding to CCW, matching the HTML tool's ccw(P) exactly.

    ccw(P) = area(P) > 0 ? P : P.slice().reverse() — a full reversal,
    which also moves index 0 to the old last vertex. Callers must
    record `flipped` and use it to remap edge indices back to the
    original (pre-normalization) vertex order wherever edges are
    re-displayed against the source geometry (設計書 2.1.2 注意).

    Returns (normalized_points, flipped).
    """
    pts = list(pts)
    s = signed_area(pts)
    if s > 0:
        return pts, False
    return pts[::-1], True


def polygon_centroid(pts: Sequence[Point]) -> Point:
    """Area-weighted centroid (not the vertex average)."""
    a = signed_area(pts)
    n = len(pts)
    if abs(a) < 1e-12:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (sum(xs) / n, sum(ys) / n)
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    factor = 1.0 / (6.0 * a)
    return (cx * factor, cy * factor)


def remove_consecutive_duplicates(pts: Sequence[Point], tol: float = 1e-9) -> List[Point]:
    out: List[Point] = []
    for p in pts:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > tol:
            out.append(p)
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


def _perp_distance(p: Point, a: Point, b: Point) -> float:
    if a == b:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg_len = math.hypot(dx, dy)
    return abs(dx * (a[1] - p[1]) - (a[0] - p[0]) * dy) / seg_len


def _douglas_peucker(pts: List[Point], eps: float) -> List[Point]:
    if len(pts) < 3:
        return pts
    dmax, index = 0.0, 0
    for i in range(1, len(pts) - 1):
        d = _perp_distance(pts[i], pts[0], pts[-1])
        if d > dmax:
            dmax, index = d, i
    if dmax > eps:
        left = _douglas_peucker(pts[: index + 1], eps)
        right = _douglas_peucker(pts[index:], eps)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def simplify_polygon(pts: Sequence[Point], tolerance: float = 0.1) -> List[Point]:
    """Douglas-Peucker simplification of a closed ring.

    The start vertex (index 0) is always kept, matching 設計書 2.1.2's
    rule that the start point must never move. Falls back to the
    original points if simplification would leave fewer than 3
    vertices (degenerate polygon).
    """
    pts = remove_consecutive_duplicates(pts)
    if len(pts) < 4:
        return list(pts)
    loop = list(pts) + [pts[0]]
    simplified = _douglas_peucker(loop, tolerance)
    if len(simplified) > 1 and simplified[-1] == simplified[0]:
        simplified = simplified[:-1]
    if len(simplified) < 3:
        return list(pts)
    return simplified


def area_change_ratio(before: Sequence[Point], after: Sequence[Point]) -> float:
    """Relative area change, e.g. 0.006 == 0.6%% (設計書 2.1.1 の 0.5%% 警告用)."""
    a0 = polygon_area(before)
    a1 = polygon_area(after)
    if a0 <= 0:
        return 0.0
    return abs(a1 - a0) / a0


def largest_ring(rings: Sequence[Sequence[Point]]) -> Tuple[List[Point], int, int]:
    """Pick the largest-area ring among candidates (multipolygon parts).

    Returns (chosen_ring, chosen_index, discarded_count).
    """
    if not rings:
        raise ValueError("no rings given")
    areas = [polygon_area(r) for r in rings]
    best = max(range(len(rings)), key=lambda i: areas[i])
    return list(rings[best]), best, len(rings) - 1


def polygon_edges(pts: Sequence[Point]) -> List[dict]:
    """Edge list matching 設計書 2.1.2's numbering: edge i = vertex i -> i+1, 1-indexed.

    `bearing` is the compass bearing (degrees clockwise from north,
    [0, 360)) that edge *faces* — i.e. its outward normal n=(dy,-dx)
    (設計書2.1.2), not the direction you'd walk along the edge. This
    matches how a boundary edge's orientation is normally described in
    practice (e.g. "南道路" for a south-facing road frontage: the edge
    itself runs east-west, but it "faces" south). Using the edge's own
    travel direction instead would be off by 90° from that intuition.
    For the UI's 方位 column only — it is not part of the exported JSON.
    """
    n = len(pts)
    edges = []
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        bearing = math.degrees(math.atan2(dy, -dx)) % 360.0
        edges.append({"index": i + 1, "length": length, "bearing": bearing})
    return edges


def m_per_deg(lat_deg: float) -> Tuple[float, float]:
    """Meters per degree of latitude/longitude at lat_deg.

    Same formula as the HTML tool's mPerDeg(lat), so JSON round-trips
    to an identical local XY (設計書 2.4).
    Returns (m_per_deg_lat, m_per_deg_lng).
    """
    p = math.radians(lat_deg)
    m_lat = 111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p)
    m_lng = 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p)
    return m_lat, m_lng


def latlng_to_local(lat: float, lng: float, origin_lat: float, origin_lng: float) -> Point:
    """Equivalent of the HTML tool's xy(): lat/lng -> local meters around origin."""
    m_lat, m_lng = m_per_deg(origin_lat)
    return ((lng - origin_lng) * m_lng, (lat - origin_lat) * m_lat)


def local_to_latlng(x: float, y: float, origin_lat: float, origin_lng: float) -> Point:
    """Equivalent of the HTML tool's ll(): local meters -> lat/lng around origin."""
    m_lat, m_lng = m_per_deg(origin_lat)
    return (origin_lat + y / m_lat, origin_lng + x / m_lng)


# ---------------------------------------------------------------------
# 探索アルゴリズム用の追加ジオメトリ (設計書 3章)。
# 以下はすべて pts が CCW 正規化済みであることを前提とする
# （呼び出し側で to_ccw 済みのものを渡すこと。HTML側は offsetPolyPer
# 内部で毎回 ccw() をかけ直すが、本コードベースでは to_ccw を呼ぶ場所を
# 明示的に1箇所に揃える方針にしている）。
# ---------------------------------------------------------------------


def edge_vector(pts: Sequence[Point], i: int) -> Tuple[Point, Point, Point, Point]:
    """辺iの始点・終点・単位方向・外向き単位法線 (n=(dy,-dx)/|d|、設計書2.1.2)。"""
    n = len(pts)
    a, b = pts[i], pts[(i + 1) % n]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    direction = (dx / length, dy / length)
    normal = (dy / length, -dx / length)
    return a, b, direction, normal


def inward_distance(p: Point, edge_a: Point, normal: Point) -> float:
    """辺の境界線（直線）からpまでの、内向きを正とする符号付き距離 (HTMLの inwardDist)。"""
    return -((p[0] - edge_a[0]) * normal[0] + (p[1] - edge_a[1]) * normal[1])


def offset_polygon_per_edge(pts: Sequence[Point], offsets: Sequence[float]) -> List[Point]:
    """辺ごとに異なる距離で外向きオフセットする（HTMLの offsetPolyPer と同一アルゴリズム）。

    offsets[i] は辺i（pts[i]->pts[i+1]）のオフセット量。正で外向き、負で
    内向き。ミター結合（隣接する2本のオフセット線の交点）で頂点を求める
    ため、鋭角な凹みでは交点が遠くへ飛ぶことがある（HTML版と同じ制限。
    設計書8.3の「既知の限界」に準ずる簡易実装）。
    """
    n = len(pts)
    lines = []
    for i in range(n):
        a, b, _direction, normal = edge_vector(pts, i)
        t = offsets[i] if i < len(offsets) else 0.0
        lines.append(
            ((a[0] + normal[0] * t, a[1] + normal[1] * t), (b[0] + normal[0] * t, b[1] + normal[1] * t))
        )
    out = []
    for i in range(n):
        p = lines[(i - 1) % n]
        q = lines[i]
        d1x, d1y = p[1][0] - p[0][0], p[1][1] - p[0][1]
        d2x, d2y = q[1][0] - q[0][0], q[1][1] - q[0][1]
        den = d1x * d2y - d1y * d2x
        if abs(den) < 1e-9:
            out.append(q[0])
            continue
        s = ((q[0][0] - p[0][0]) * d2y - (q[0][1] - p[0][1]) * d2x) / den
        out.append((p[0][0] + s * d1x, p[0][1] + s * d1y))
    return out


def rectangle_corners(cx: float, cy: float, W: float, D: float, rot_deg: float) -> List[Point]:
    """建物矩形の4隅（HTMLの corners(b) と同一規則。W方向がu軸、D方向がv軸）。"""
    rot = math.radians(rot_deg)
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    hw, hd = W / 2.0, D / 2.0
    out = []
    for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        out.append((cx + su * hw * cos_r + sv * hd * sin_r, cy - su * hw * sin_r + sv * hd * cos_r))
    return out


def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    """点1個の内外判定（レイキャスティング、HTMLの inPoly と同一）。"""
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            denom = (yj - yi) if yj != yi else 1e-12
            if x < (xj - xi) * (y - yi) / denom + xi:
                inside = not inside
        j = i
    return inside


def point_in_polygon_mask(X: np.ndarray, Y: np.ndarray, poly: Sequence[Point]) -> np.ndarray:
    """point_in_polygon のnumpyベクトル化版（格子点をまとめて判定する）。"""
    inside = np.zeros(np.broadcast(X, Y).shape, dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if yi != yj:
            denom = yj - yi
            cond = ((yi > Y) != (yj > Y)) & (X < (xj - xi) * (Y - yi) / denom + xi)
            inside ^= cond
        j = i
    return inside


def point_to_polygon_distance(X: np.ndarray, Y: np.ndarray, poly: Sequence[Point]) -> np.ndarray:
    """各格子点から多角形の外周までの最短距離（HTMLの polyDist のベクトル化版）。"""
    n = len(poly)
    min_d = None
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            px = np.full(np.broadcast(X, Y).shape, ax)
            py = np.full(np.broadcast(X, Y).shape, ay)
        else:
            t = ((X - ax) * dx + (Y - ay) * dy) / L2
            t = np.clip(t, 0.0, 1.0)
            px, py = ax + t * dx, ay + t * dy
        d = np.hypot(X - px, Y - py)
        min_d = d if min_d is None else np.minimum(min_d, d)
    return min_d


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and d1 != 0 and d2 != 0 and ((d3 > 0) != (d4 > 0)) and d3 != 0 and d4 != 0:
        return True

    def on_seg(a: Point, p: Point, b: Point) -> bool:
        return (
            min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9
        )

    if abs(d1) < 1e-9 and on_seg(p3, p1, p4):
        return True
    if abs(d2) < 1e-9 and on_seg(p3, p2, p4):
        return True
    if abs(d3) < 1e-9 and on_seg(p1, p3, p2):
        return True
    if abs(d4) < 1e-9 and on_seg(p1, p4, p2):
        return True
    return False


def rectangle_in_polygon(
    center: Point, u: Point, v: Point, half_w: float, half_h: float, poly: Sequence[Point]
) -> bool:
    """回転した矩形（中心center、u/v単位軸、半幅half_w、半奥行half_h）がpoly内に完全に収まるか。

    4隅が内部にあるだけでなく、矩形の辺がpolyの辺と交差しないことも
    確認する（凹んだ切れ込みが4隅を避けて矩形内に食い込むケースの対策）。
    """
    cx, cy = center
    ux, uy = u
    vx, vy = v
    corners = []
    for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        corners.append((cx + su * half_w * ux + sv * half_h * vx, cy + su * half_w * uy + sv * half_h * vy))
    for c in corners:
        if not point_in_polygon(c, poly):
            return False
    n = len(poly)
    for i in range(4):
        r1, r2 = corners[i], corners[(i + 1) % 4]
        for j in range(n):
            p1, p2 = poly[j], poly[(j + 1) % n]
            if _segments_intersect(r1, r2, p1, p2):
                return False
    return True


def candidate_seed_points(poly: Sequence[Point], k: int = 4, grid_steps: int = 60) -> List[Point]:
    """poly内部の「局所的に外周から遠い」点を複数返す（内接長方形探索の起点候補）。

    敷地がL字・段状など非凸な場合、単一の代表点（重心や大域的な最遠点）
    だけでは片方の腕にしか長方形を作れないことがある
    （rectangle_for_aspect_ratio は中心をseedに固定して探索するため）。
    境界ボックスをグリッドスキャンし、外周までの距離が大きい点から
    非最大抑制で間引きながら上位k個を選ぶことで、複数の腕それぞれに
    候補点を持たせる。呼び出し側（core.search）は各候補で内接長方形を
    求め、最も面積の大きいものを採用する。
    """
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 <= x0 or y1 <= y0:
        return []
    gx = np.linspace(x0, x1, grid_steps)
    gy = np.linspace(y0, y1, grid_steps)
    X, Y = np.meshgrid(gx, gy)
    inside = point_in_polygon_mask(X, Y, poly)
    if not inside.any():
        return []
    dist = point_to_polygon_distance(X, Y, poly)
    dist = np.where(inside, dist, -1.0)

    suppress_radius = max(x1 - x0, y1 - y0) / max(4, k * 2)
    flat_d = dist.ravel()
    flat_x = X.ravel()
    flat_y = Y.ravel()
    order = np.argsort(-flat_d)

    picked: List[Point] = []
    for idx in order:
        if flat_d[idx] <= 0:
            break
        x, y = float(flat_x[idx]), float(flat_y[idx])
        if any(math.hypot(x - px, y - py) < suppress_radius for px, py in picked):
            continue
        picked.append((x, y))
        if len(picked) >= k:
            break
    return picked


def interior_seed_point(poly: Sequence[Point]) -> Optional[Point]:
    """poly内部の点を1つ返す（内接長方形探索の起点用の簡易版）。

    まず面積重心を試し（凸多角形やおおむね凸に近い形状ならこれで十分）、
    内部でなければ candidate_seed_points の最上位点にフォールバックする。
    複数腕を持つ非凸敷地を扱うときは、この関数ではなく
    candidate_seed_points を使うこと。
    """
    c = polygon_centroid(poly)
    if point_in_polygon(c, poly):
        return c
    pts = candidate_seed_points(poly, k=1)
    return pts[0] if pts else None


def rectangle_for_aspect_ratio(
    poly: Sequence[Point],
    rotation_deg: float,
    aspect_ratio: float,
    seed: Optional[Point] = None,
    tol: float = 0.02,
) -> Optional[Dict[str, float]]:
    """指定回転角・縦横比(width/depth)で、中心をseedに固定したまま内接する最大サイズを求める。

    設計書3.3の3-4番「回転角ごとに内接する最大長方形を求め、縦横比を
    数段階に振る」を1つの関数にまとめたもの: 比率ごとにこの関数を呼べば、
    正方形寄り〜細長い形までの候補列がそのまま得られる。中心を固定して
    スケールだけを二分探索するので、4辺を独立に伸ばす一般的な最大内接
    長方形探索より単純で高速（探索は多数の候補を評価する前提なので、
    1候補ごとの厳密な最適性より候補生成の速さを優先している）。

    Returns {"cx","cy","W","D","rot","area"} or None if nothing fits.
    """
    rot = math.radians(rotation_deg)
    # u,v はW軸・D軸の単位ベクトル。rectangle_corners()（HTMLのcorners(b)を
    # 移植したもの、W軸=(cos,-sin)・D軸=(sin,cos)という時計回り正の規則）と
    # 必ず同じ向きにすること。ここをCCW正の(cos,sin)/(-sin,cos)にすると、
    # 「fitする」と判定した矩形と、rectangle_corners()で実際に使われる
    # 矩形が回転の向きだけ鏡映（rot⇔-rot）になり、fit判定は通ったのに
    # 実際の矩形は敷地からの離隔が確保できていない、という食い違いが
    # 起きる（rot=0/180度では鏡映しても同じ矩形になるため気づきにくい）。
    u = (math.cos(rot), -math.sin(rot))
    v = (math.sin(rot), math.cos(rot))
    if seed is None:
        seed = interior_seed_point(poly)
    if seed is None or not point_in_polygon(seed, poly):
        return None
    sx, sy = seed

    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    max_extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) + 1.0

    half_w0 = aspect_ratio / 2.0
    half_h0 = 0.5

    def fits(scale: float) -> bool:
        if scale <= 1e-9:
            return True
        return rectangle_in_polygon((sx, sy), u, v, half_w0 * scale, half_h0 * scale, poly)

    lo, hi = 0.0, max_extent
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if fits(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    if lo <= 0:
        return None
    W, D = half_w0 * 2 * lo, half_h0 * 2 * lo
    return {"cx": sx, "cy": sy, "W": W, "D": D, "rot": rotation_deg, "area": W * D}
