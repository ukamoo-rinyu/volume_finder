"""最大ボリューム探索アルゴリズム（純粋Python/numpy、QGIS非依存、設計書3章）。

位置・回転・幅・奥行・階数の5次元を総当たりせず、二重構造で絞り込む
（設計書3.1-3.2）:

    外側ループ: 平面形の候補（位置・回転・幅・奥行）
        -> 斜線・容積率・建蔽率から最大階数を閉じた式で逆算（一瞬）
        -> 枝刈り: この上限でも現在の上位N件を超えないなら評価しない
    内側ループ: 日影規制を満たす最大階数を二分探索（対象のときだけ）

段階5（複数用途地域）: 設計書3.6により、容積率・建蔽率は敷地全体で
面積按分した値（法52条7項・法53条2項、呼び出し側が prorate 済みの
zone/far/bcr を渡す）を使う一方、用途地域そのもの・斜線・日影規制は
按分できず「区分ごとにその区分の基準で判定」する。本モジュールでは
これを、候補ごとの重心がどの区分の領域に入っているかで判定する
簡易的な実装にしている（`zone_regions`、法91条の「大きい部分の属する
地域の規定による」という考え方を、重心所属という安価な近似に単純化
したもの）。単純な直方体1棟のみという設計書8.3の制約は変わらない。
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import geometry as geo
from . import regulation as reg
from . import shadow as sh

Point = Tuple[float, float]


@dataclass
class SearchParams:
    min_setback: float = 1.0  # 最小離隔 (設計書3.3-1)。buildable_override指定時は無視される。
    buildable_override: Optional[Sequence[Point]] = None
    """min_setbackぶん内側にオフセットした建築可能範囲を、呼び出し側が
    あらかじめ計算して渡す場合に使う。core.geometry.offset_polygon_per_edge
    は辺ごとのミター結合による簡易実装で、凹んだ敷地（鋭い切れ込みや
    細い部分がある形）では正しくオフセットできず、指定した最小離隔より
    実際には敷地境界に近い場所に建物が置かれてしまうことがある
    （単体テストでは軸並行の矩形など素直な形しか使わないため気づき
    にくい）。QGIS側ではGEOSの堅牢なbuffer()が使えるので、ui/dock.pyは
    これで最小離隔ぶん内側にオフセットした範囲を計算してここに渡す。
    Noneならoffset_polygon_per_edge(site_pts, [-min_setback]*n)にフォール
    バックする（単体テストなど、素直な凸形状での簡易利用向け）。"""
    floor_height: float = 3.5  # 階高
    max_floors: int = 20  # 階数上限（探索条件のUI入力）
    rotation_step_deg: float = 15.0  # 回転角の刻み（辺方向は常に含む）
    aspect_ratios: Sequence[float] = field(
        default_factory=lambda: (0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0)
    )
    shrink_factors: Sequence[float] = field(default_factory=lambda: (1.0, 0.85, 0.7))
    shadow_dt_minutes: float = 5.0
    shadow_cell: Optional[float] = None
    bcr_relax: str = "0"  # "0"|"10"|"10f"|"20" (設計書、core.regulation.bcr_limit)
    max_results: int = 3
    min_footprint_dim: float = 1.0  # これより小さい辺を持つ候補は捨てる
    hikage_override: Optional[Dict[str, float]] = None
    """設計書3.5「影が落ちる先の用途地域」。{"mh","t1","t2"} を指定すると、
    敷地自身の用途地域が日影規制の対象外でも、この基準で日影を判定する
    （呼び出し側が敷地の周辺A29から最も厳しい基準を求めて渡す）。
    Noneなら従来どおり zone/far から core.regulation.hikage_rule を引く。"""
    zone_regions: Optional[Sequence[dict]] = None
    """設計書3.6「複数用途地域」用。[{"zone_key","far","polygon":[(x,y),...]}, ...]。
    site_pts と同じ座標系（EPSG:6674メートル）の、敷地をA29の区分で
    分割した領域。候補の重心がどの領域に入るかで、その候補に適用する
    zone/far を決める（斜線・日影規制の判定に使う。容積率・建蔽率は
    このパラメータに関わらず search() に渡された敷地全体の按分値を
    使い続ける）。Noneまたは重心がどの領域にも入らない場合は、
    search() に渡された zone/far にフォールバックする（単一用途地域の
    段階3の挙動と同じ）。"""


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
    rank: int = 0


def _rotation_candidates(site_pts: Sequence[Point], step_deg: float) -> List[float]:
    """回転角の候補: 敷地の各辺の方向 + step_deg刻み（設計書3.3-2）。

    長方形は90度ごとに同じ形の集合を作るので、[0,90)の範囲だけで十分。
    """
    angles = set()
    n = len(site_pts)
    for i in range(n):
        _a, _b, direction, _normal = geo.edge_vector(site_pts, i)
        deg = math.degrees(math.atan2(direction[1], direction[0])) % 90.0
        angles.add(round(deg, 3))
    step = max(1.0, step_deg)
    a = 0.0
    while a < 90.0 - 1e-6:
        angles.add(round(a, 3))
        a += step
    return sorted(angles)


def _footprint_candidates(buildable: Sequence[Point], site_pts: Sequence[Point], params: SearchParams) -> List[dict]:
    """回転・縦横比・縮小段階を組み合わせて平面形の候補を作る（設計書3.3-2〜5）。"""
    rotations = _rotation_candidates(site_pts, params.rotation_step_deg)
    results: List[dict] = []
    for rot in rotations:
        seeds = geo.candidate_seed_points(buildable, k=4)
        if not seeds:
            continue
        for ratio in params.aspect_ratios:
            best = None
            for seed in seeds:
                rect = geo.rectangle_for_aspect_ratio(buildable, rot, ratio, seed=seed)
                if rect and (best is None or rect["area"] > best["area"]):
                    best = rect
            if best is None or best["W"] <= 0 or best["D"] <= 0:
                continue
            for shrink in params.shrink_factors:
                W, D = best["W"] * shrink, best["D"] * shrink
                if W < params.min_footprint_dim or D < params.min_footprint_dim:
                    continue
                results.append({"cx": best["cx"], "cy": best["cy"], "W": W, "D": D, "rot": rot})
    return results


def _local_zone(
    cx: float, cy: float, zone_regions: Optional[Sequence[dict]], fallback_zone: str, fallback_far: float
) -> Tuple[str, float]:
    """候補の重心がどの用途地域区分に属するかを決める（設計書3.6）。

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
    """道路斜線・隣地斜線から許容高さを閉じた式で逆算する（設計書3.1、HTMLの slopeCheck の核心部分）。

    Returns (許容高さの最小値[m]（辺なしなら inf）, その高さを与えた辺のラベル一覧)。
    """
    n = len(site_pts)
    limits: List[Tuple[str, float]] = []
    for i in range(n):
        edge = edges[i] if i < len(edges) else {"type": "rinchi", "width": 0}
        a, _b, _direction, normal = geo.edge_vector(site_pts, i)
        A = max(0.0, min(geo.inward_distance(c, a, normal) for c in corners))
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
    hikage_override: Optional[Dict[str, float]] = None,
) -> Tuple[int, Optional[dict]]:
    """日影規制を満たす最大階数を二分探索する（設計書3.2・3.4）。

    まず上限階数で判定し、通ればそれで確定（二分探索不要）。落ちた場合
    だけ [0, upper_bound) の範囲で二分探索する（設計書3.4）。

    hikage_override が与えられていればそれを使う（設計書3.5「影が落ちる
    先の用途地域」: 敷地自身の zone/far が対象外でも、周辺の対象区域の
    基準が適用されることがある）。測定面高さ(mh)は規制自体に紐づく値
    なので、常に採用した規則（override または zone/far 由来）の mh を
    使う（HTMLの syncLevels() が $("mh") を HR.mh に自動同期するのと同じ
    扱い。段階3のような手動mh入力は持たない）。
    """
    rule = hikage_override if hikage_override is not None else reg.hikage_rule(zone, far)
    if rule is None or upper_bound <= 0:
        return upper_bound, None

    def passes(floors: int) -> bool:
        if floors <= 0:
            return True
        H = floors * floor_height
        bands = sh.shadow_bands(
            footprint["cx"], footprint["cy"], footprint["W"], footprint["D"], footprint["rot"], H,
            deemed_boundary, mh=rule["mh"], dt_minutes=dt_minutes, cell=cell,
        )
        return bands["max_5_10"] < rule["t1"] - 1e-6 and bands["max_10plus"] < rule["t2"] - 1e-6

    if passes(upper_bound):
        return upper_bound, rule

    lo, hi = 0, upper_bound  # lo は常に「適合する」側、hi は「超過する」側
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if passes(mid):
            lo = mid
        else:
            hi = mid
    return lo, rule


def search(
    site_pts: Sequence[Point],
    edges: Sequence[dict],
    zone: str,
    far: float,
    bcr: float,
    params: Optional[SearchParams] = None,
    progress_callback: Optional[Callable[[int, int, Optional[Candidate]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[Candidate]:
    """最大ボリュームの上位候補を探索する（設計書3章、段階3）。

    site_pts: CCW正規化済み・EPSG:6674などメートル単位の敷地座標。
    edges: site_pts[i]->[i+1] に対応する [{"type","width"}, ...]。
    zone/far/bcr: 単一用途地域を想定（複数用途地域の按分は段階5）。
    progress_callback(done, total, current_best): 中断可能なUIから
    QgsTask経由で呼ばれる想定（設計書4.2）。current_bestはNoneのことも
    ある。should_stop(): Trueを返すとその時点までの結果を返して打ち切る
    （設計書4.2「中断時も、その時点の最良案を返す」）。
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

    footprints = _footprint_candidates(buildable, site_pts, params)
    total = len(footprints)
    best_list: List[Candidate] = []

    def worst_kept_area() -> float:
        if len(best_list) < params.max_results:
            return -1.0  # 上位N件がまだ埋まっていなければ、面積に関わらず必ず評価する
        return best_list[-1].floor_area

    def report(idx: int) -> None:
        if progress_callback:
            progress_callback(idx, total, best_list[0] if best_list else None)

    for idx, fp in enumerate(footprints, start=1):
        if should_stop and should_stop():
            break
        footprint_area = fp["W"] * fp["D"]

        bcr_ok, _bcr_lim = _bcr_ok(footprint_area, site_area, bcr, params.bcr_relax)
        if not bcr_ok:
            report(idx)
            continue

        # 容積率(・前面道路幅員による制限)は敷地全体の按分値のまま (設計書3.6、法52条7項)。
        floors_far, _far_lim, _wmax = _far_floor_limit(footprint_area, site_area, zone, far, edges)
        floors_far = min(floors_far, params.max_floors)

        # 用途地域・斜線は按分できず、候補の重心が属する区分の基準で判定する
        # (設計書3.6、zone_regions未指定なら従来どおり敷地全体のzone/farを使う)。
        local_zone, local_far = _local_zone(fp["cx"], fp["cy"], params.zone_regions, zone, far)

        corners = geo.rectangle_corners(fp["cx"], fp["cy"], fp["W"], fp["D"], fp["rot"])
        slant_h, slant_binding = _slant_max_height(corners, site_pts, edges, local_zone, local_far)
        floors_slant = (
            min(params.max_floors, int(math.floor(slant_h / params.floor_height + 1e-9)))
            if math.isfinite(slant_h)
            else params.max_floors
        )

        upper_bound = min(floors_far, floors_slant, params.max_floors)
        if upper_bound <= 0:
            report(idx)
            continue

        # 枝刈り: 上限階数でも現在の上位N件の最下位を超えないなら日影計算をしない (設計書3.2)
        if footprint_area * upper_bound <= worst_kept_area() + 1e-6:
            report(idx)
            continue

        floors_final, hikage_rule_used = _floors_within_shadow(
            fp, local_zone, local_far, deemed_boundary, params.floor_height,
            upper_bound, params.shadow_dt_minutes, params.shadow_cell,
            hikage_override=params.hikage_override,
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
        if hikage_rule_used is not None and floors_final < upper_bound:
            binding.append("日影規制（影が落ちる先）" if params.hikage_override is not None else f"日影規制{zone_suffix}")
        if not binding:
            binding.append("容積率")

        floor_area = footprint_area * floors_final
        candidate = Candidate(
            cx=fp["cx"], cy=fp["cy"], W=fp["W"], D=fp["D"], rot=fp["rot"],
            floors=floors_final, height=floors_final * params.floor_height,
            footprint_area=footprint_area, floor_area=floor_area,
            far_pct=floor_area / site_area * 100, bcr_pct=footprint_area / site_area * 100,
            binding=binding,
        )
        best_list.append(candidate)
        best_list.sort(key=lambda c: -c.floor_area)
        del best_list[params.max_results:]
        report(idx)

    for rank, c in enumerate(best_list, start=1):
        c.rank = rank
    return best_list
