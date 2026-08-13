"""日影規制の計算（純粋Python/numpy、QGIS非依存、設計書 3.4）。

既存HTMLツールの sun()/offsetAt()/inShadow()/computeIso() と同じ
アルゴリズムをnumpyでベクトル化して移植する。探索（core.search）では
1つの建物配置あたり繰り返し評価されるため、HTMLツールの高精細グリッド
（長辺480分割・既定2分刻み）よりも粗い既定解像度を使う。

設計書 8.2「探索結果の検証」は、探索で得た案をJSONで書き出して
HTMLツールで再計算し、両者の判定が一致することを確認する運用を
前提としており、解像度差による差異はそこで吸収する想定。
"""

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from . import geometry as geo

_DEG = math.pi / 180.0
DECL_WINTER_SOLSTICE = -23.45 * _DEG  # 冬至日の太陽赤緯
HOUR_START, HOUR_END = 8.0, 16.0  # 真太陽時 8時〜16時 (設計書8.1)


def sun_position(hour: float, phi_deg: float = 35.0, decl: float = DECL_WINTER_SOLSTICE) -> Optional[Tuple[float, float]]:
    """(方位角[rad], 倍率) を返す。太陽が地平線下ならNone (HTMLの sun() と同一)。

    倍率は「有効高さ×倍率＝影の長さ」の係数 (= cos(高度)/sin(高度))。
    設計書8.1の検証値: 北緯35度・冬至日 8時で 方位角−53°27′・倍率6.71。
    """
    phi = phi_deg * _DEG
    t = (hour - 12) * 15 * _DEG
    sh = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(t)
    if sh <= 1e-6:
        return None
    ch = math.sqrt(1 - sh * sh)
    az = math.atan2(
        math.cos(decl) * math.sin(t) / ch,
        (sh * math.sin(phi) - math.sin(decl)) / (ch * math.cos(phi)),
    )
    return az, ch / sh


def _shadow_mask(U: np.ndarray, V: np.ndarray, ou: float, ov: float, hw: float, hd: float) -> np.ndarray:
    """建物ローカル座標系での「箱＋オフセット線分のミンコフスキー和」内判定 (HTMLの inShadow と同一)。

    ou/ov は建物ローカル座標系での日影オフセット（1時刻あたりのスカラー
    値）なので、ここではPython分岐のみで済み、格子側はnumpyで一括処理する。
    """
    s0 = np.zeros_like(U)
    s1 = np.ones_like(U)
    u_ok = None
    v_ok = None
    if ou == 0:
        u_ok = np.abs(U) <= hw
    else:
        a, c = (U - hw) / ou, (U + hw) / ou
        lo, hi = np.minimum(a, c), np.maximum(a, c)
        s0 = np.maximum(s0, lo)
        s1 = np.minimum(s1, hi)
    if ov == 0:
        v_ok = np.abs(V) <= hd
    else:
        a, c = (V - hd) / ov, (V + hd) / ov
        lo, hi = np.minimum(a, c), np.maximum(a, c)
        s0 = np.maximum(s0, lo)
        s1 = np.minimum(s1, hi)
    mask = s0 <= s1
    if u_ok is not None:
        mask &= u_ok
    if v_ok is not None:
        mask &= v_ok
    return mask


def max_shadow_multiplier(phi_deg: float = 35.0, step_hours: float = 0.1) -> float:
    """時間帯8-16時のうちで最大となる影の倍率（8時・16時付近が最大）。グリッド余白の見積もり用。"""
    best = 0.0
    hour = HOUR_START
    while hour <= HOUR_END + 1e-9:
        sp = sun_position(hour, phi_deg)
        if sp:
            best = max(best, sp[1])
        hour += step_hours
    return best


def shadow_reach_distance(effective_height: float, phi_deg: float = 35.0) -> float:
    """影が到達しうる最大距離の見積り（設計書3.5「影が落ちる先の用途地域」用）。

    有効高さ×最大倍率（冬至日8時・16時付近が最大、北緯35度で約6.71倍）。
    実際の影は方位ごとに長さが変わる扇形だが、設計書3.5-1の「有効高さ×
    6.71」という単純な定義に合わせ、全方位に対する円形バッファとして
    安全側（広め）に見積もる。この距離でA29を検索し、影が実際に届く
    かどうかに関わらず「対象になりうる区域」を洗い出すのに使う。
    """
    if effective_height <= 0:
        return 0.0
    return effective_height * max_shadow_multiplier(phi_deg)


def shadow_raster(
    cx: float,
    cy: float,
    W: float,
    D: float,
    rot_deg: float,
    H: float,
    deemed_boundary: Sequence[Tuple[float, float]],
    mh: float = 4.0,
    dt_minutes: float = 5.0,
    cell: Optional[float] = None,
    max_grid_span: float = 150.0,
    phi_deg: float = 35.0,
) -> Optional[dict]:
    """建物1棟について、日影時間のラスタ（HTMLの computeIso() の中核）を計算する。

    段階3は「単純な直方体1棟のみ」という設計書8.3の既知の限界と一致する
    スコープなので、複数棟の日影重なり処理（HTML版のmark配列による
    タイムステップ内重複排除）は省いている。

    deemed_boundary: 日影規制のみなし境界線（敷地形状を辺ごとの
    core.regulation.hikage_relax 分だけ外側にオフセットしたもの。
    core.geometry.offset_polygon_per_edge で作る）。

    Returns {"X","Y" (格子座標), "total" (各セルの日影時間h), "dist"
    (みなし境界線までの距離), "measurable" (みなし境界線の外＝測定対象)}。
    測定面より建物が低ければ None（影が生じない）。

    bands_from_raster() と組み合わせて使う: ラスタ自体（時刻ループを含む、
    最も重い部分）は測定面高さ(mh)ごとに1回だけ計算し、複数の区域
    （影が落ちる先の区域ごとに異なるマスク範囲・規制時間）の判定に
    使い回すことで、区域数ぶんラスタ計算を繰り返さずに済む
    （設計書3.6・3.5、複数の日影規制区域を個別に判定するための土台）。
    """
    eff_top = H - mh
    if eff_top <= 0:
        return None

    rot = rot_deg * _DEG
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    hw, hd = W / 2.0, D / 2.0

    xs = [p[0] for p in deemed_boundary]
    ys = [p[1] for p in deemed_boundary]
    bx0, bx1 = min(xs), max(xs)
    by0, by1 = min(ys), max(ys)

    worst_mult = max_shadow_multiplier(phi_deg)
    margin = eff_top * worst_mult + max(hw, hd) + 4.0

    x0 = min(bx0, cx - hw) - margin
    x1 = max(bx1, cx + hw) + margin
    y0 = min(by0, cy - hd) - margin
    y1 = max(by1, cy + hd) + margin

    span = max(x1 - x0, y1 - y0)
    if cell is None:
        cell = max(1.0, span / max_grid_span)
    nx = max(1, int(math.ceil((x1 - x0) / cell)))
    ny = max(1, int(math.ceil((y1 - y0) / cell)))

    xs_grid = x0 + (np.arange(nx) + 0.5) * cell
    ys_grid = y0 + (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs_grid, ys_grid)  # shape (ny, nx)

    gx, gy = X - cx, Y - cy
    U = gx * cos_r - gy * sin_r
    Vloc = gx * sin_r + gy * cos_r

    inside = geo.point_in_polygon_mask(X, Y, deemed_boundary)
    dist = geo.point_to_polygon_distance(X, Y, deemed_boundary)
    measurable = ~inside

    steps = max(1, round((HOUR_END - HOUR_START) * 60 / dt_minutes))
    dt_h = dt_minutes / 60.0
    total = np.zeros((ny, nx), dtype=np.float64)
    for k in range(steps):
        hour = HOUR_START + (k + 0.5) * dt_minutes / 60.0
        sp = sun_position(hour, phi_deg)
        if sp is None:
            continue
        az, mult = sp
        L2 = eff_top * mult
        ex, ey = L2 * math.sin(az), L2 * math.cos(az)
        ou = ex * cos_r - ey * sin_r
        ov = ex * sin_r + ey * cos_r
        mask = _shadow_mask(U, Vloc, ou, ov, hw, hd)
        total += mask * dt_h

    return {"X": X, "Y": Y, "total": total, "dist": dist, "measurable": measurable}


def bands_from_raster(raster: Optional[dict], mask_polygon: Optional[Sequence[Tuple[float, float]]] = None) -> dict:
    """ラスタ（shadow_rasterの戻り値）から、5-10m帯・10m超帯の最大日影時間(h)を取り出す。

    mask_polygon を指定すると、そのポリゴンの内側にあるセルだけを対象に
    する。「影が落ちる先の区域」（法56条の2第4項）の規制は、影が実際に
    その区域の土地に落ちている部分にしか適用されないため、対象区域の
    ポリゴンで絞り込んでから最大日影時間を求める必要がある（区域の外の
    土地に落ちる影の長さでその区域の規制を判定してはいけない）。

    Returns {"max_5_10": float, "max_10plus": float} (時間, h)。
    """
    if raster is None:
        return {"max_5_10": 0.0, "max_10plus": 0.0}
    measurable = raster["measurable"]
    if mask_polygon:
        measurable = measurable & geo.point_in_polygon_mask(raster["X"], raster["Y"], mask_polygon)
    dist = raster["dist"]
    total = raster["total"]
    band1 = total[measurable & (dist > 5) & (dist <= 10)]
    band2 = total[measurable & (dist > 10)]
    max1 = float(band1.max()) if band1.size else 0.0
    max2 = float(band2.max()) if band2.size else 0.0
    return {"max_5_10": max1, "max_10plus": max2}


def shadow_bands(
    cx: float,
    cy: float,
    W: float,
    D: float,
    rot_deg: float,
    H: float,
    deemed_boundary: Sequence[Tuple[float, float]],
    mh: float = 4.0,
    dt_minutes: float = 5.0,
    cell: Optional[float] = None,
    max_grid_span: float = 150.0,
    phi_deg: float = 35.0,
) -> dict:
    """建物1棟について、敷地全周（マスクなし）の5-10m帯・10m超帯の
    最大日影時間(h)を返す（shadow_raster + bands_from_raster の単純合成）。

    敷地自身の用途地域が日影規制の対象のときに使う（法56条の2の原則:
    規制対象区域内の建築物からの影は、受ける側の土地の用途地域を問わず
    規制される）。「影が落ちる先の区域」判定のように区域ごとにマスクが
    必要な場合は、shadow_raster()を1回呼んでから bands_from_raster()を
    区域の数だけ呼ぶこと（時刻ループの計算を使い回せる）。
    """
    raster = shadow_raster(cx, cy, W, D, rot_deg, H, deemed_boundary, mh, dt_minutes, cell, max_grid_span, phi_deg)
    return bands_from_raster(raster)
