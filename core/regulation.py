"""大阪市の建築規制の基準値（純粋Python、QGIS非依存）。

既存HTMLツール (hikage-osaka-v4_23.html) の値と完全に一致させる
（設計書 8.1 検証方針）。段階2で必要な日影規制の対象区分判定に加え、
段階3の探索で使う道路斜線・隣地斜線・前面道路幅員による容積率制限・
建蔽率の緩和・日影のみなし境界線の緩和値を持つ。
"""

from typing import Dict, Optional, Sequence, Tuple

# 住居系用途地域（HTMLの JUKYO 配列と同一）。
JUKYO = ["1中高", "2中高", "1住", "2住", "準住"]

# HTMLの ZNAME と同一（プラグイン内部で使う10区分の正式名称）。
ZNAME: Dict[str, str] = {
    "1中高": "第1種中高層住専",
    "2中高": "第2種中高層住専",
    "1住": "第1種住居",
    "2住": "第2種住居",
    "準住": "準住居",
    "近商": "近隣商業",
    "商業": "商業",
    "準工": "準工業",
    "工業": "工業",
    "工専": "工業専用",
}

# 大阪市の日影規制6区分（HTMLの hikageRule() と同一の値）。
# (zone, far) -> {mh, t1(5-10m), t2(10m超)}
_HIKAGE_TABLE: Dict[Tuple[str, int], Dict[str, float]] = {
    ("1中高", 200): {"mh": 4, "t1": 4, "t2": 2.5},
    ("2中高", 300): {"mh": 4, "t1": 5, "t2": 3},
    ("1住", 200): {"mh": 4, "t1": 5, "t2": 3},
    ("2住", 200): {"mh": 4, "t1": 5, "t2": 3},
    ("準住", 200): {"mh": 4, "t1": 5, "t2": 3},
    ("準工", 200): {"mh": 6.5, "t1": 5, "t2": 3},
}


def hikage_rule(zone: str, far: float) -> Optional[Dict[str, float]]:
    """(用途地域, 指定容積率) から大阪市の日影規制値を返す。対象外なら None。

    HTMLの hikageRule(z,f) と同じ丸め: far は int 化して照合する。
    """
    return _HIKAGE_TABLE.get((zone, int(far)))


def strictest_hikage(candidates: Sequence[Tuple[str, float]]) -> Optional[Tuple[str, float, Dict[str, float]]]:
    """複数区分にまたがる敷地で、日影規制の対象となる区分のうち最も厳しいものを選ぶ。

    設計書 3.6「用途地域・日影規制・斜線：按分不可。区分ごとにその区分の
    基準で判定」を受けて、按分ではなく「最も厳しい基準を採用」という
    単純化を行う（設計書 4.1 モックアップの「⚠ 日影規制は厳しい側を採用」
    に対応）。「厳しさ」は、10m超区画の許容時間 t2 が小さいほど厳しいと
    定義し、同点なら 5-10m区画の t1 で比較する。この順序付けは実際の
    日影シミュレーション結果を保証するものではない簡易的な目安であり、
    段階3以降で区分ごとの詳細判定に置き換わる。

    candidates: [(zone_key, far), ...]
    Returns (zone_key, far, rule) の最も厳しい組、対象が無ければ None。
    """
    scored = []
    for zone, far in candidates:
        rule = hikage_rule(zone, far)
        if rule is not None:
            scored.append((zone, far, rule))
    if not scored:
        return None
    return min(scored, key=lambda item: (item[2]["t2"], item[2]["t1"]))


# ---------------------------------------------------------------------
# 斜線・容積率・建蔽率 (設計書 3章、段階3の探索で使う)
# ---------------------------------------------------------------------


def road_rule(zone: str, far: float) -> Dict[str, object]:
    """前面道路の道路斜線の勾配・適用距離 (HTMLの roadRule(z,f) と同一)。

    Returns {"k": 勾配の分母(1:k), "L": 適用距離(m), "kind": 系統名}。
    """
    if zone in JUKYO:
        L = 20 if far <= 200 else 25 if far <= 300 else 30 if far <= 400 else 35
        return {"k": 1.25, "L": L, "kind": "住居系"}
    if zone in ("近商", "商業"):
        L = (
            20 if far <= 400 else 25 if far <= 600 else 30 if far <= 800
            else 35 if far <= 1000 else 40 if far <= 1100 else 45 if far <= 1200 else 50
        )
        return {"k": 1.5, "L": L, "kind": "商業系"}
    L = 20 if far <= 200 else 25 if far <= 300 else 30 if far <= 400 else 35
    return {"k": 1.5, "L": L, "kind": "工業系"}


def rin_rule(zone: str) -> Dict[str, float]:
    """隣地斜線の立ち上がり高さ・勾配 (HTMLの rinRule(z) と同一)。

    Returns {"H0": 立ち上がり高さ(m), "k": 勾配の分母(1:k)}。
    """
    return {"H0": 20, "k": 1.25} if zone in JUKYO else {"H0": 31, "k": 2.5}


def far_coef(zone: str) -> float:
    """前面道路幅員による容積率制限の係数 (HTMLの farCoef(z) と同一)。"""
    return 0.4 if zone in JUKYO else 0.6


def bcr_limit(base_bcr: float, relax: str = "0") -> float:
    """建蔽率の限度 (HTMLの bcrLimit() と同一)。

    relax: "0"=なし, "10"=角地+10%, "10f"=防火地域+耐火+10%,
    "20"=角地+防火地域+耐火+20%。指定建蔽率80%かつ"10f"は無制限。
    """
    if base_bcr == 80 and relax == "10f":
        return float("inf")
    bonus = {"0": 0, "10": 10, "10f": 10, "20": 20}.get(relax, 0)
    return base_bcr + bonus


# 日影のみなし境界線の緩和対象（HTMLの ETYPE[*].hikage===true な3種）。
HIKAGE_RELAX_TYPES = {"road", "sui", "koen"}


def hikage_relax(edge_type: str, width: float) -> float:
    """日影規制のみなし境界線の緩和量 (設計書2.3、HTMLの hikageRelax と同一)。

    道路・水面線路敷・公園広場に接する辺は、幅員の半分（10m以下）または
    幅員-5m（10m超）だけ外側にみなし境界線を引ける。隣地は対象外で0。
    """
    if edge_type not in HIKAGE_RELAX_TYPES:
        return 0.0
    w = width or 0.0
    if w <= 0:
        return 0.0
    return w / 2.0 if w <= 10 else w - 5.0
