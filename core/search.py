"""最大ボリューム探索アルゴリズム（純粋Python/numpy、QGIS非依存）。

ver0.2.0（volume-study-search-spec.md）で、容積最大化のみの単一目的探索から、
実務的な配置を導く多目的探索に変更した。位置・回転・幅・奥行・階数の5次元を
総当たりせず、二重構造で絞り込む点は変わらない:

    外側ループ: 平面形の候補（回転・アンカー位置・幅・奥行、仕様1-2章）
        -> 斜線・容積率・建蔽率から最大階数を閉じた式で逆算（一瞬）
        -> 枝刈り: この上限でもプールの最下位を超えないなら評価しない
    内側ループ: 日影規制を満たす最大階数を二分探索（対象のときだけ）
        -> プール（floor_area上位pool_size件）に暫定採用
    プール確定後: 有効空地・南面採光などの多目的スコアを計算し、
        min-max正規化 → 重み付き総合スコアでランキング → 類似解を間引く
        （仕様3.7・6章）。重み変更時はプールを使って rerank() するだけで
        再探索は不要。

段階5（複数用途地域）: 容積率・建蔽率は敷地全体で面積按分した値
（法52条7項・法53条2項、呼び出し側が prorate 済みの zone/far/bcr を渡す）を
使う一方、用途地域そのもの・斜線・日影規制は按分できず「区分ごとにその
区分の基準で判定」する。本モジュールでは、候補ごとの重心がどの区分の
領域に入っているかで判定する簡易実装にしている（`zone_regions`、法91条の
「大きい部分の属する地域の規定による」という考え方を、重心所属という
安価な近似に単純化したもの）。

単純な直方体1棟のみという制約は ver0.2.0 でも変えていない
（L型・コの字型・分棟などの配置類型別探索とパレート散布図UIは、
実装量に対して既存の枝刈り・日影計算の多くを拡張し直す必要があり
見送った。詳細は volume-study-search-spec.md §5, §7 と、このリポジトリの
QGISプラグイン設計書8.3「既知の限界」を参照）。
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import geometry as geo
from . import regulation as reg
from . import shadow as sh

Point = Tuple[float, float]

ANCHOR_KEYS: Tuple[str, ...] = ("N", "S", "E", "W", "NE", "NW", "SE", "SW", "C")
ANCHOR_LABELS: Dict[str, str] = {
    "N": "北寄せ", "S": "南寄せ", "E": "東寄せ", "W": "西寄せ",
    "NE": "北東寄せ", "NW": "北西寄せ", "SE": "南東寄せ", "SW": "南西寄せ", "C": "中央",
}


@dataclass
class SearchParams:
    min_setback: float = 1.0  # 最小離隔。buildable_override指定時は無視される。
    buildable_override: Optional[Sequence[Point]] = None
    """min_setbackぶん内側にオフセットした建築可能範囲を、呼び出し側が
    あらかじめ計算して渡す場合に使う。core.geometry.offset_polygon_per_edge
    は辺ごとのミター結合による簡易実装で、凹んだ敷地（鋭い切れ込みや
    細い部分がある形）では正しくオフセットできず、指定した最小離隔より
    実際には敷地境界に近い場所に建物が置かれてしまうことがある。
    QGIS側ではGEOSの堅牢なbuffer()が使えるので、ui/dock.pyはこれで
    最小離隔ぶん内側にオフセットした範囲を計算してここに渡す。
    Noneならoffset_polygon_per_edge(site_pts, [-min_setback]*n)にフォール
    バックする（単体テストなど、素直な凸形状での簡易利用向け）。"""
    floor_height: float = 3.5  # 階高
    max_floors: int = 20  # 階数上限（探索条件のUI入力）

    rotation_mode: str = "edge"  # "edge"|"orthogonal"|"free" (仕様1章)
    rotation_step_deg: float = 15.0  # rotation_mode=="free" のときだけ使う刻み

    size_fractions: Sequence[float] = field(default_factory=lambda: (1.0, 0.85, 0.7, 0.55, 0.4))
    """建築可能範囲のローカル座標系での外接矩形 [Umin,Umax]×[Vmin,Vmax] に
    対する幅・奥行の縮小率。W,Dそれぞれ独立にこの集合から選ぶので、
    比率の異なる組み合わせが自然に生成される（仕様3.3の「縦横比を数段階
    に振る」＋「縮小方向へ数段階振る」を1つのパラメータにまとめたもの）。"""
    anchors: Sequence[str] = field(default_factory=lambda: ANCHOR_KEYS)
    """使うアンカーの集合（仕様2章）。北/南/東/西は自由軸を持つ。"""
    anchor_slide_step: float = 1.0  # 自由軸のスライド刻み(m)（仕様2章は一律1.0m）
    anchor_slide_max_positions: int = 25
    """自由軸のスライド候補数の上限。敷地が広いと1.0m刻みでは候補数が
    爆発するため、この上限を超えないよう自動的にstepを広げる
    （仕様10章の「候補数が10万を超える場合は寸法刻みを粗くする」と同趣旨の
    実務的な調整）。"""

    open_space_cell: float = 0.5  # 有効空地ラスタのグリッド間隔(m)（仕様3.2）

    weight_far: float = 50.0  # 容積消化率の重み（仕様3.7）
    weight_open: float = 30.0  # 有効空地の重み
    weight_light: float = 5.0  # 南面採光の重み
    weight_impact: float = 0.0  # 隣地影響の重み（0なら計算自体を省略、仕様3.6はオプション）
    """整形度(S_shape)・施工性(S_simple)は仕様3.3/3.5にあるが、単純な
    直方体1棟のみという制約下ではどの候補も必ず1.0になり（外形＝外接矩形、
    セットバック段数・棟数の変動が無いため）、min-max正規化すると
    候補間で常に同値＝順位に一切影響しないので、計算自体を割愛している。"""

    shadow_dt_minutes: float = 5.0
    shadow_cell: Optional[float] = None
    bcr_relax: str = "0"  # "0"|"10"|"10f"|"20" (core.regulation.bcr_limit)

    pool_size: int = 400
    """floor_areaの大きい順に保持する候補プールの上限。多目的スコアの
    min-max正規化は本来「適合候補すべて」から取るべきだが（仕様3.7）、
    敷地全体を舐める候補（size_fractions×anchors×rotationsで数千〜）
    すべてに有効空地のラスタ計算をかけると探索が遅くなるため、floor_area
    で絞ったプールで近似する（実務上の割り切り。値を上げるほど厳密に
    近づくが遅くなる）。"""
    max_results: int = 8  # 類似解間引き後に返す件数（仕様6章のN=8が既定）
    dedupe_threshold: float = 10.0  # 類似解間引きの距離しきい値(m)（仕様6章）

    min_footprint_dim: float = 1.0  # これより小さい辺を持つ候補は捨てる
    slant_safety_margin_m: float = 0.05
    """斜線制限（道路斜線・隣地斜線）の許容高さから、階数を決める前に
    差し引く安全余裕(m)。JSON書き出し・再読み込み（緯度経度への往復変換、
    座標のmm丸め）を経てHTMLツールで再計算すると、後退距離Aがmm〜cm単位
    でずれうるため、境界ぎりぎりの階数を選んでしまうとHTML側の再判定で
    「超過」に反転することがある（実際に見つかった不具合）。この余裕分は
    ほとんどの場合1階分にも満たないため、選べる階数への影響はごく小さい。"""
    hikage_regions: Optional[Sequence[dict]] = None
    """「影が落ちる先の用途地域」（法56条の2第4項）用の追加区域。
    [{"mask": [(x,y),...] または None, "rule": {"mh","t1","t2"}}, ...]。

    敷地自身の用途地域（candidateごとのlocal_zone/local_far）が日影規制の
    対象なら、そちらの基準は"mask"なし（全周に適用、法56条の2の原則:
    規制対象区域内の建築物からの影は、受ける側の土地の用途地域を問わず
    規制される）で自動的に追加されるので、ここに重ねて書く必要はない。
    ここに渡すのは、敷地の外側にある別区画（周辺A29の対象区域）の基準
    だけで、必ずその区画の実際のポリゴン（EPSG:6674、site_ptsと同じ
    座標系）を"mask"として指定すること。"mask"を省略/Noneにすると
    全周に適用されてしまい、影が実際には届かない方角の区域まで
    誤って制約に含めてしまうので注意（ver0.2.0で修正した不具合。
    単体テストなどマスクを用意しない簡易利用のときだけNoneを使う）。
    候補は、ここに含めたすべての区域＋（該当すれば）敷地自身の区域の
    基準を、それぞれの適用範囲内ですべて満たす必要がある。"""
    zone_regions: Optional[Sequence[dict]] = None
    """複数用途地域用。[{"zone_key","far","polygon":[(x,y),...]}, ...]。
    site_pts と同じ座標系（EPSG:6674メートル）の、敷地をA29の区分で
    分割した領域。候補の重心がどの領域に入るかで、その候補に適用する
    zone/far を決める（斜線・日影規制の判定に使う。容積率・建蔽率は
    このパラメータに関わらず search() に渡された敷地全体の按分値を
    使い続ける）。Noneまたは重心がどの領域にも入らない場合は、
    search() に渡された zone/far にフォールバックする。"""


@dataclass
class Candidate:
    cx: float
    cy: float
    W: float
    D: float
    rot: float
    floors: int
    height: float
    footprint_area: float
    floor_area: float
    far_pct: float
    bcr_pct: float
    binding: List[str]
    anchor: str = ""
    far_ratio: float = 0.0  # S_far = floor_area / 法定延床面積上限 (仕様3.1)
    open_area: float = 0.0  # 有効空地面積(m²)（仕様3.2、参考表示用の生値）
    open_min_side: float = 0.0  # 有効空地の短辺(m)
    light_ratio: float = 0.0  # S_light (仕様3.4)
    impact: float = 1.0  # S_impact (仕様3.6。weight_impact>0のときだけ実測)
    norm: Dict[str, float] = field(default_factory=dict)  # min-max正規化後の各指標
    score: float = 0.0  # 重み付き総合スコア（仕様3.7）
    rank: int = 0


# ---------------------------------------------------------------------
# 回転角の候補 (仕様1章)
# ---------------------------------------------------------------------


def _edge_length(site_pts: Sequence[Point], i: int) -> float:
    a, b, _direction, _normal = geo.edge_vector(site_pts, i)
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _edge_angles_mod90(site_pts: Sequence[Point]) -> List[float]:
    angles = []
    for i in range(len(site_pts)):
        _a, _b, direction, _normal = geo.edge_vector(site_pts, i)
        angles.append(math.degrees(math.atan2(direction[1], direction[0])) % 90.0)
    return angles


def _dedupe_angles_mod90(angles: Sequence[float], tol_deg: float = 1.0) -> List[float]:
    """1度以内は同一とみなして重複除去する（仕様1章）。"""
    if not angles:
        return []
    uniq = sorted(set(round(a % 90.0, 6) for a in angles))
    out = [uniq[0]]
    for a in uniq[1:]:
        if a - out[-1] > tol_deg:
            out.append(a)
    if len(out) > 1 and (90.0 - out[-1]) + out[0] <= tol_deg:
        out.pop()
    return out


def _main_road_edge_index(site_pts: Sequence[Point], edges: Sequence[dict]) -> Optional[int]:
    """主要接道辺: 幅員最大の道路辺（同値なら辺長最大）。道路辺が無ければNone。"""
    candidates = []
    for i, e in enumerate(edges):
        if e.get("type") != "road":
            continue
        width = float(e.get("width") or 0)
        if width <= 0:
            continue
        candidates.append((width, _edge_length(site_pts, i), i))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (-t[0], -t[1]))
    return candidates[0][2]


def _rotation_candidates(
    site_pts: Sequence[Point],
    edges: Sequence[dict],
    mode: str = "edge",
    step_deg: float = 15.0,
) -> List[float]:
    """回転角の候補 (仕様1章)。

    "edge"（既定）: 敷地の各辺の方向のみ（mod 90、1度以内は同一とみなし
        重複除去）。建物は必ずいずれかの境界線に平行に建てるのが実務、
        という前提で「斜め配置」の候補をそもそも作らない。
    "orthogonal": 主要接道辺（幅員最大の道路辺、無ければ最長辺）の方向のみ
        の1通り。
    "free": 従来相当（辺方向 ∪ step_deg刻みの全域）。比較用に残す。
    """
    edge_angles = _edge_angles_mod90(site_pts)
    if not site_pts:
        return [0.0]

    if mode == "orthogonal":
        idx = _main_road_edge_index(site_pts, edges)
        if idx is None:
            idx = max(range(len(site_pts)), key=lambda i: _edge_length(site_pts, i))
        return _dedupe_angles_mod90([edge_angles[idx]])

    angles = list(edge_angles) + [0.0]
    if mode == "free":
        step = max(1.0, step_deg)
        a = 0.0
        while a < 90.0 - 1e-6:
            angles.append(a)
            a += step
    return _dedupe_angles_mod90(angles)


# ---------------------------------------------------------------------
# アンカー配置 (仕様2章)
# ---------------------------------------------------------------------


def _anchor_positions(
    umin: float, umax: float, vmin: float, vmax: float, W: float, D: float,
    anchor: str, slide_step: float, max_slide_positions: int,
) -> List[Tuple[float, float]]:
    """アンカーごとの建物左下座標(u0,v0)の候補一覧（ローカルu/v座標系、仕様2章の表）。

    北=+V方向, 南=-V方向, 東=+U方向, 西=-U方向（rot=0でu=world x, v=world y、
    core.geometry.local_bounding_box/local_to_world_center 参照）。北/南/東/西
    は自由軸を持つため、slide_stepごとにスライドさせた候補を複数返す。
    W,Dがbboxに収まらなければ空リスト。
    """
    span_u = umax - umin - W
    span_v = vmax - vmin - D
    if span_u < -1e-9 or span_v < -1e-9:
        return []

    def slide(lo: float, span: float) -> List[float]:
        if span <= 1e-9:
            return [lo]
        step = max(slide_step, span / max(1, max_slide_positions))
        n = int(math.floor(span / step + 1e-9)) + 1
        return [lo + min(k * step, span) for k in range(n)]

    if anchor == "N":
        return [(u0, vmax - D) for u0 in slide(umin, span_u)]
    if anchor == "S":
        return [(u0, vmin) for u0 in slide(umin, span_u)]
    if anchor == "E":
        return [(umax - W, v0) for v0 in slide(vmin, span_v)]
    if anchor == "W":
        return [(umin, v0) for v0 in slide(vmin, span_v)]
    if anchor == "NE":
        return [(umax - W, vmax - D)]
    if anchor == "NW":
        return [(umin, vmax - D)]
    if anchor == "SE":
        return [(umax - W, vmin)]
    if anchor == "SW":
        return [(umin, vmin)]
    if anchor == "C":
        return [((umin + umax - W) / 2.0, (vmin + vmax - D) / 2.0)]
    return []


def _footprint_candidates(buildable: Sequence[Point], params: SearchParams, rotations: Sequence[float]) -> List[dict]:
    """回転・寸法・アンカーを組み合わせて平面形の候補を作る（仕様1-2章）。

    アンカーで決めた位置に置いた矩形が建築可能領域からはみ出す場合は、
    押し戻さずに候補ごと破棄する（仕様2章「アンカーの意味が消えるため」）。
    """
    results: List[dict] = []
    seen = set()
    for rot in rotations:
        umin, umax, vmin, vmax = geo.local_bounding_box(buildable, rot)
        span_u, span_v = umax - umin, vmax - vmin
        if span_u <= 0 or span_v <= 0:
            continue
        rot_rad = math.radians(rot)
        cos_r, sin_r = math.cos(rot_rad), math.sin(rot_rad)
        u_axis = (cos_r, -sin_r)
        v_axis = (sin_r, cos_r)
        for fu in params.size_fractions:
            W = span_u * fu
            if W < params.min_footprint_dim:
                continue
            for fv in params.size_fractions:
                D = span_v * fv
                if D < params.min_footprint_dim:
                    continue
                for anchor in params.anchors:
                    for u0, v0 in _anchor_positions(
                        umin, umax, vmin, vmax, W, D, anchor,
                        params.anchor_slide_step, params.anchor_slide_max_positions,
                    ):
                        cx, cy = geo.local_to_world_center(u0 + W / 2.0, v0 + D / 2.0, rot)
                        key = (rot, round(cx, 3), round(cy, 3), round(W, 3), round(D, 3))
                        if key in seen:
                            continue
                        # アンカー配置は建物の辺が建築可能領域の境界にちょうど
                        # 乗ることが多い（それがアンカーの目的）。point_in_polygon
                        # は境界上の点を内外どちらとも判定しうる（レイキャスティング
                        # の境界条件依存）ため、判定だけわずかに内側に縮めた矩形で
                        # 行い、浮動小数点誤差で境界一致の候補が誤って弾かれるのを防ぐ。
                        eps = 1e-4
                        if not geo.rectangle_in_polygon(
                            (cx, cy), u_axis, v_axis, W / 2.0 - eps, D / 2.0 - eps, buildable
                        ):
                            continue
                        seen.add(key)
                        results.append({"cx": cx, "cy": cy, "W": W, "D": D, "rot": rot, "anchor": anchor})
    return results


# ---------------------------------------------------------------------
# 適法性判定 (建蔽率・容積率・斜線・日影)
# ---------------------------------------------------------------------


def _road_width_max(edges: Sequence[dict]) -> float:
    widths = [float(e.get("width") or 0) for e in edges if e.get("type") == "road"]
    return max(widths) if widths else 0.0


def _far_limit(zone: str, far: float, edges: Sequence[dict]) -> Tuple[float, float]:
    """前面道路幅員による容積率制限を適用した基準容積率 (HTMLの judgeAll() と同一ロジック)。

    道路の辺が無い場合はHTMLと同じく基準容積率0%になる
    （接道していない敷地は実質的に建築不可、という制約をそのまま反映）。
    """
    wmax = _road_width_max(edges)
    if wmax >= 12:
        return far, wmax
    far_road = wmax * reg.far_coef(zone) * 100
    return min(far, far_road), wmax


def _bcr_ok(footprint_area: float, site_area: float, bcr: float, bcr_relax: str) -> Tuple[bool, float]:
    limit = reg.bcr_limit(bcr, bcr_relax)
    if math.isinf(limit):
        return True, limit
    return footprint_area / site_area * 100 <= limit + 1e-6, limit


def _far_floor_limit(footprint_area: float, site_area: float, zone: str, far: float, edges: Sequence[dict]):
    far_lim, wmax = _far_limit(zone, far, edges)
    if far_lim <= 0 or footprint_area <= 0:
        return 0, far_lim, wmax
    floor_area_limit = far_lim / 100.0 * site_area
    return int(math.floor(floor_area_limit / footprint_area + 1e-9)), far_lim, wmax


def _slant_max_height(
    corners: Sequence[Point], site_pts: Sequence[Point], edges: Sequence[dict], zone: str, far: float
) -> Tuple[float, List[str]]:
    """道路斜線・隣地斜線から許容高さを閉じた式で逆算する（HTMLの slopeCheck の核心部分）。

    Returns (許容高さの最小値[m]（辺なしなら inf）, その高さを与えた辺のラベル一覧)。
    """
    n = len(site_pts)
    limits: List[Tuple[str, float]] = []
    for i in range(n):
        edge = edges[i] if i < len(edges) else {"type": "rinchi", "width": 0}
        a, b, direction, normal = geo.edge_vector(site_pts, i)
        A = max(0.0, min(geo.inward_distance(c, a, b, direction, normal) for c in corners))
        etype = edge.get("type", "rinchi")
        width = float(edge.get("width") or 0)
        if etype == "road" and width > 0:
            R = reg.road_rule(zone, far)
            d_start = width + 2 * A
            if d_start > R["L"]:
                continue  # 適用距離を超えるため道路斜線の制限なし（HTMLの beyond=true と同じ扱い）
            limits.append((f"道路斜線(辺{i + 1})", R["k"] * d_start))
        else:
            if etype == "road":
                continue  # 幅員未入力の道路辺は道路斜線・隣地斜線どちらの対象にもしない(HTML同様)
            RR = reg.rin_rule(zone)
            a2 = A + (width / 2.0 if etype in ("sui", "koen") else 0.0)
            limits.append((f"隣地斜線(辺{i + 1})", RR["H0"] + RR["k"] * 2 * a2))
    if not limits:
        return float("inf"), []
    min_height = min(h for _label, h in limits)
    binding = [label for label, h in limits if h <= min_height + 1e-6]
    return min_height, binding


def _floors_within_shadow(
    footprint: dict,
    zone: str,
    far: float,
    deemed_boundary: Sequence[Point],
    floor_height: float,
    upper_bound: int,
    dt_minutes: float,
    cell: Optional[float],
    hikage_regions: Optional[Sequence[dict]] = None,
) -> Tuple[int, List[dict]]:
    """日影規制を満たす最大階数を二分探索する。

    まず上限階数で判定し、通ればそれで確定（二分探索不要）。落ちた場合
    だけ [0, upper_bound) の範囲で二分探索する。

    敷地自身の(zone, far)が日影規制の対象なら、その基準は敷地の全周
    （mask なし）に適用する（法56条の2の原則: 規制対象区域内の建築物
    からの影は、受ける側の土地の用途地域を問わず規制される）。
    hikage_regions（「影が落ちる先の用途地域」、法56条の2第4項）は
    それぞれの mask（対象区域のポリゴン。Noneなら全周）内でだけ判定する
    ——対象区域の外の土地に落ちる影の長さで、その区域の規制を判定しては
    いけない（ver0.2.0で修正した不具合: 到達範囲内に対象区域があるだけで
    全方位に規制をかけてしまい、実際には影が届かない方角まで不必要に
    制約していた）。

    測定面高さ(mh)が異なる区域が混在する場合、日影ラスタ（時刻ループを
    含む最も重い計算）はmhごとに1回だけ計算し、同じmhの区域どうしで
    使い回す（shadow_raster/bands_from_raster参照）。

    Returns (floors_final, binding_regions)。binding_regionsは
    floors_final+1階では満たせなかった区域（{"kind":"own"|"target",
    "name":Optional[str]}）。規制自体が無い、または規制が階数を
    制限しなかった（floors_final==upper_bound）場合は空リスト。
    """
    own_rule = reg.hikage_rule(zone, far)
    regions: List[dict] = []
    if own_rule is not None:
        regions.append({"mask": None, "rule": own_rule, "kind": "own", "name": None})
    for r in hikage_regions or []:
        regions.append({"mask": r.get("mask"), "rule": r["rule"], "kind": "target", "name": r.get("name")})
    if not regions or upper_bound <= 0:
        return upper_bound, []

    mh_groups: Dict[float, List[dict]] = {}
    for region in regions:
        mh_groups.setdefault(region["rule"]["mh"], []).append(region)

    def failing_regions(floors: int) -> List[dict]:
        if floors <= 0:
            return []
        H = floors * floor_height
        failed = []
        for mh, group in mh_groups.items():
            raster = sh.shadow_raster(
                footprint["cx"], footprint["cy"], footprint["W"], footprint["D"], footprint["rot"], H,
                deemed_boundary, mh=mh, dt_minutes=dt_minutes, cell=cell,
            )
            for region in group:
                bands = sh.bands_from_raster(raster, region["mask"])
                rule = region["rule"]
                if not (bands["max_5_10"] < rule["t1"] - 1e-6 and bands["max_10plus"] < rule["t2"] - 1e-6):
                    failed.append(region)
        return failed

    if not failing_regions(upper_bound):
        return upper_bound, []

    lo, hi = 0, upper_bound  # lo は常に「適合する」側、hi は「超過する」側
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if not failing_regions(mid):
            lo = mid
        else:
            hi = mid
    binding = failing_regions(lo + 1) if lo < upper_bound else []
    return lo, [{"kind": r["kind"], "name": r["name"]} for r in binding]


def _floors_from_max_height(max_height: float, floor_height: float, max_floors: int) -> int:
    """許容高さから階数を逆算する（斜線制限用。math.isfiniteでなければ階数上限まで許容）。"""
    if not math.isfinite(max_height):
        return max_floors
    return min(max_floors, int(math.floor(max_height / floor_height + 1e-9)))


def _local_zone(
    cx: float, cy: float, zone_regions: Optional[Sequence[dict]], fallback_zone: str, fallback_far: float
) -> Tuple[str, float]:
    """候補の重心がどの用途地域区分に属するかを決める。

    zone_regions内の、重心を含む領域のzone/farを返す。重なりがある
    場合は先勝ち（呼び出し側で面積の大きい順に並べておくことを想定）。
    どの領域にも入らなければ fallback（敷地全体で選ばれている用途地域）
    を返す。
    """
    if not zone_regions:
        return fallback_zone, fallback_far
    for region in zone_regions:
        polygon = region.get("polygon")
        if polygon and geo.point_in_polygon((cx, cy), polygon):
            return region["zone_key"], region["far"]
    return fallback_zone, fallback_far


# ---------------------------------------------------------------------
# 多目的スコア (仕様3章) と類似解の間引き (仕様6章)
# ---------------------------------------------------------------------


def _south_wall_length(corners: Sequence[Point]) -> float:
    """南面（真南±45°を向く外壁）の壁長さの合計 (仕様3.4のS_light分子)。"""
    total = 0.0
    n = len(corners)
    for i in range(n):
        a, b = corners[i], corners[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        bearing = math.degrees(math.atan2(dy, -dx)) % 360.0
        diff = min(abs(bearing - 180.0), 360.0 - abs(bearing - 180.0))
        if diff <= 45.0 + 1e-6:
            total += length
    return total


def _minmax_normalize(values: Sequence[float]) -> List[float]:
    """指標ごとのmin-max正規化 (仕様3.7)。max==minなら全候補1.0。"""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _weighted_score(norm: Dict[str, float], weights: Dict[str, float]) -> float:
    total_w = sum(max(0.0, w) for w in weights.values())
    if total_w <= 0:
        return 0.0
    return sum(max(0.0, weights.get(k, 0.0)) * norm.get(k, 0.0) for k in weights) / total_w


def _dedupe_similar(
    candidates: Sequence[Candidate], site_area: float, threshold: float, limit: int
) -> List[Candidate]:
    """類似解の間引き (仕様6章)。candidatesはスコア降順に並んでいる前提。"""

    def distance(a: Candidate, b: Candidate) -> float:
        d = abs(a.cx - b.cx) + abs(a.cy - b.cy)
        if site_area > 0:
            d += 0.5 * abs(a.floor_area - b.floor_area) / site_area
        if abs(a.rot - b.rot) > 1e-6:
            d += 999.0
        return d

    kept: List[Candidate] = []
    for c in candidates:
        if all(distance(c, k) >= threshold for k in kept):
            kept.append(c)
        if len(kept) >= limit:
            break
    return kept


def rerank(
    pool: Sequence[Candidate],
    weights: Dict[str, float],
    site_area: float,
    dedupe_threshold: float = 10.0,
    max_results: int = 8,
) -> List[Candidate]:
    """重み変更時の再ランキング（仕様3.7「重み変更時はソートし直すだけ」）。

    pool は search() が pool_out 引数経由で返す候補プール（各要素の
    .norm に正規化済みの値が入っている）。生の指標を再計算せず、
    正規化済みの値から重み付き総合スコアを出し直してソート・類似解間引き
    をするだけなので、UIのスライダー操作に対して瞬時に反映できる。
    """
    scored = list(pool)
    for c in scored:
        c.score = _weighted_score(c.norm, weights)
    scored.sort(key=lambda c: -c.score)
    kept = _dedupe_similar(scored, site_area, dedupe_threshold, max_results)
    for rank, c in enumerate(kept, start=1):
        c.rank = rank
    return kept


def search(
    site_pts: Sequence[Point],
    edges: Sequence[dict],
    zone: str,
    far: float,
    bcr: float,
    params: Optional[SearchParams] = None,
    progress_callback: Optional[Callable[[int, int, Optional[Candidate]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    pool_out: Optional[List[Candidate]] = None,
) -> List[Candidate]:
    """実用的な配置の上位候補を探索する（仕様1-3・6章）。

    site_pts: CCW正規化済み・EPSG:6674などメートル単位の敷地座標。
    edges: site_pts[i]->[i+1] に対応する [{"type","width"}, ...]。
    zone/far/bcr: 単一用途地域を想定（複数用途地域の按分は呼び出し側で
    prorate 済みの値を渡す）。
    progress_callback(done, total, current_best): 中断可能なUIから
    QgsTask経由で呼ばれる想定。current_bestはNoneのこともある。
    should_stop(): Trueを返すとその時点までの結果を返して打ち切る
    （中断時も、その時点の最良案を返す）。
    pool_out: 渡すと、多目的スコア計算後・類似解間引き前の候補プール
    （スコア降順）で満たされる。UI側がこれを保持しておけば、重みを
    変えるたびに rerank() を呼ぶだけで再探索なしに並べ替えられる。
    """
    params = params or SearchParams()
    site_area = geo.polygon_area(site_pts)
    if site_area <= 0:
        return []

    if params.buildable_override is not None:
        buildable = list(params.buildable_override)
    else:
        buildable = geo.offset_polygon_per_edge(site_pts, [-params.min_setback] * len(site_pts))
    if not buildable or geo.polygon_area(buildable) <= 1.0:
        return []

    relax_offsets = [reg.hikage_relax(e.get("type", "rinchi"), float(e.get("width") or 0)) for e in edges]
    deemed_boundary = geo.offset_polygon_per_edge(site_pts, relax_offsets)

    rotations = _rotation_candidates(site_pts, edges, params.rotation_mode, params.rotation_step_deg)
    footprints = _footprint_candidates(buildable, params, rotations)
    total = len(footprints)
    pool: List[Candidate] = []

    def worst_kept_area() -> float:
        if len(pool) < params.pool_size:
            return -1.0  # プールがまだ埋まっていなければ、面積に関わらず必ず評価する
        return pool[-1].floor_area

    def report(idx: int) -> None:
        if progress_callback:
            progress_callback(idx, total, pool[0] if pool else None)

    for idx, fp in enumerate(footprints, start=1):
        if should_stop and should_stop():
            break
        footprint_area = fp["W"] * fp["D"]

        bcr_ok, _bcr_lim = _bcr_ok(footprint_area, site_area, bcr, params.bcr_relax)
        if not bcr_ok:
            report(idx)
            continue

        # 容積率(・前面道路幅員による制限)は敷地全体の按分値のまま (法52条7項)。
        floors_far, far_lim, _wmax = _far_floor_limit(footprint_area, site_area, zone, far, edges)
        floors_far = min(floors_far, params.max_floors)
        far_floor_area_limit = far_lim / 100.0 * site_area if far_lim > 0 else 0.0

        # 用途地域・斜線は按分できず、候補の重心が属する区分の基準で判定する
        # (zone_regions未指定なら従来どおり敷地全体のzone/farを使う)。
        local_zone, local_far = _local_zone(fp["cx"], fp["cy"], params.zone_regions, zone, far)

        corners = geo.rectangle_corners(fp["cx"], fp["cy"], fp["W"], fp["D"], fp["rot"])
        slant_h, slant_binding = _slant_max_height(corners, site_pts, edges, local_zone, local_far)
        # slant_safety_margin_m だけ引いてから階数を決める。JSON書き出し
        # （緯度経度への往復変換・座標を小数第3位に丸めての保存、設計書2.4）
        # を経てHTMLツールで再計算すると、後退距離Aがmm〜cm単位でずれう得る
        # （二重の丸め込みの角の位置がわずかに動くため）。斜線の許容高さは
        # A×斜線勾配で効くため、境界ぎりぎり（許容高さちょうど）の階数を
        # 選んでしまうと、この程度のずれだけでHTML側の再計算で「超過」に
        # 反転してしまうことがある（実際に見つかった不具合）。設計書8.2の
        # 「探索結果をJSON書き出し→HTMLツールで再計算→両者の判定が一致する
        # ことを確認する」という運用が成立するよう、あらかじめ余裕を残す。
        slant_h_safe = slant_h - params.slant_safety_margin_m if math.isfinite(slant_h) else slant_h
        floors_slant = _floors_from_max_height(slant_h_safe, params.floor_height, params.max_floors)

        upper_bound = min(floors_far, floors_slant, params.max_floors)
        if upper_bound <= 0:
            report(idx)
            continue

        # 枝刈り: 上限階数でもプールの最下位を超えないなら日影計算をしない
        if footprint_area * upper_bound <= worst_kept_area() + 1e-6:
            report(idx)
            continue

        floors_final, hikage_binding = _floors_within_shadow(
            fp, local_zone, local_far, deemed_boundary, params.floor_height,
            upper_bound, params.shadow_dt_minutes, params.shadow_cell,
            hikage_regions=params.hikage_regions,
        )
        if floors_final <= 0:
            report(idx)
            continue

        zone_suffix = f"[{reg.ZNAME.get(local_zone, local_zone)}]" if local_zone != zone else ""
        binding: List[str] = []
        if floors_final == floors_far:
            binding.append("容積率")
        if floors_final == floors_slant:
            binding.extend(f"{label}{zone_suffix}" for label in slant_binding)
        if floors_final == params.max_floors:
            binding.append("階数上限")
        for hb in hikage_binding:
            if hb["kind"] == "own":
                binding.append(f"日影規制{zone_suffix}")
            else:
                name_part = f"：{hb['name']}" if hb.get("name") else ""
                binding.append(f"日影規制（影が落ちる先{name_part}）")
        if not binding:
            binding.append("容積率")

        floor_area = footprint_area * floors_final
        far_ratio = min(floor_area / far_floor_area_limit, 1.0) if far_floor_area_limit > 0 else 0.0
        candidate = Candidate(
            cx=fp["cx"], cy=fp["cy"], W=fp["W"], D=fp["D"], rot=fp["rot"],
            floors=floors_final, height=floors_final * params.floor_height,
            footprint_area=footprint_area, floor_area=floor_area,
            far_pct=floor_area / site_area * 100, bcr_pct=footprint_area / site_area * 100,
            binding=binding, anchor=fp.get("anchor", ""), far_ratio=far_ratio,
        )
        pool.append(candidate)
        pool.sort(key=lambda c: -c.floor_area)
        del pool[params.pool_size:]
        report(idx)

    if not pool:
        return []

    # ---- 有効空地・南面採光・隣地影響 (仕様3.2, 3.4, 3.6) ----
    # 生の指標はプール（floor_area上位pool_size件、SearchParams.pool_size参照）
    # についてのみ計算する。正規化のmin/maxは本来「適合候補全体」から取るべき
    # だが(仕様3.7)、敷地全体を舐める候補すべてにラスタ計算をかけると遅く
    # なるため、floor_areaで絞ったプールで近似している。
    raw_open: List[float] = []
    raw_light: List[float] = []
    raw_impact: List[float] = []
    for c in pool:
        corners = geo.rectangle_corners(c.cx, c.cy, c.W, c.D, c.rot)

        open_result = geo.effective_open_space(site_pts, [corners], cell=params.open_space_cell)
        residual = site_area - c.footprint_area
        open_ratio = (open_result["area"] / residual) if residual > 1e-6 else 0.0
        if open_result["min_side"] < 4.0:  # 駐車場として使えない細い空地はペナルティ (仕様3.2補足)
            open_ratio *= 0.5
        c.open_area = open_result["area"]
        c.open_min_side = open_result["min_side"]
        raw_open.append(min(open_ratio, 1.0))

        c.light_ratio = min(_south_wall_length(corners) / (2.0 * math.sqrt(c.footprint_area)), 1.0)
        raw_light.append(c.light_ratio)

        impact = 1.0
        if params.weight_impact > 0:
            local_zone, local_far = _local_zone(c.cx, c.cy, params.zone_regions, zone, far)
            own_rule = reg.hikage_rule(local_zone, local_far)
            # S_impactは仕様上オプションの簡易指標（weight_impact既定0）なので、
            # 敷地自身の区域が対象ならそれを、対象外なら最初の影が落ちる先の
            # 区域（あれば、そのmaskで絞って）を代表値として使う簡略化。
            mask = None
            rule = own_rule
            if rule is None and params.hikage_regions:
                first = params.hikage_regions[0]
                rule = first["rule"]
                mask = first.get("mask")
            if rule:
                raster = sh.shadow_raster(
                    c.cx, c.cy, c.W, c.D, c.rot, c.height, deemed_boundary,
                    mh=rule["mh"], dt_minutes=params.shadow_dt_minutes, cell=params.shadow_cell,
                )
                bands = sh.bands_from_raster(raster, mask)
                impact = 1.0 - max(bands["max_5_10"] / rule["t1"], bands["max_10plus"] / rule["t2"])
        c.impact = max(0.0, min(impact, 1.0))
        raw_impact.append(c.impact)

    far_norm = _minmax_normalize([c.far_ratio for c in pool])
    open_norm = _minmax_normalize(raw_open)
    light_norm = _minmax_normalize(raw_light)
    impact_norm = _minmax_normalize(raw_impact)
    for c, f, o, l, im in zip(pool, far_norm, open_norm, light_norm, impact_norm):
        c.norm = {"far": f, "open": o, "light": l, "impact": im}

    weights = {
        "far": params.weight_far, "open": params.weight_open,
        "light": params.weight_light, "impact": params.weight_impact,
    }
    if pool_out is not None:
        for c in pool:
            c.score = _weighted_score(c.norm, weights)
        pool_out.extend(sorted(pool, key=lambda c: -c.score))

    return rerank(pool, weights, site_area, params.dedupe_threshold, params.max_results)
