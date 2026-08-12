"""用途地域の判定・按分ロジック（純粋Python、QGIS非依存）。

国土数値情報 A29 の読み込みと敷地との空間演算（ui/dock.py 側で実施）
から渡された「区分ごとの面積・属性」を受け取り、以下を行う:

  - コードと名称の突き合わせ（設計書 2.2）
  - 容積率・建蔽率の面積按分（設計書 3.6、法52条7項・法53条2項）
  - 按分値が容積率・建蔽率の区分境界に近いことの警告（設計書 2.2）

日影規制・道路斜線・隣地斜線は「按分不可、区分ごとにその区分の基準で
判定」（設計書 3.6）なので、どの区分に日影規制の基準があるかは
core.regulation を使う。
"""

from typing import Dict, List, Optional, Sequence, Tuple

# 設計書 2.2 の対応表（区分コード 1〜13）。
CODE_TABLE: Dict[int, Tuple[str, str]] = {
    1: ("1低層", "第一種低層住居専用地域"),
    2: ("2低層", "第二種低層住居専用地域"),
    3: ("1中高", "第一種中高層住居専用地域"),
    4: ("2中高", "第二種中高層住居専用地域"),
    5: ("1住", "第一種住居地域"),
    6: ("2住", "第二種住居地域"),
    7: ("準住", "準住居地域"),
    8: ("近商", "近隣商業地域"),
    9: ("商業", "商業地域"),
    10: ("準工", "準工業地域"),
    11: ("工業", "工業地域"),
    12: ("工専", "工業専用地域"),
    13: ("田園住居", "田園住居地域"),
}

CODE_TO_KEY = {code: key for code, (key, _name) in CODE_TABLE.items()}

# A29_005 の名称表記ゆれ（漢数字/算用数字）を両方とも許容する。
NAME_TO_KEY: Dict[str, str] = {}
for _code, (_key, _name) in CODE_TABLE.items():
    NAME_TO_KEY[_name] = _key
NAME_TO_KEY.update(
    {
        "第1種低層住居専用地域": "1低層",
        "第2種低層住居専用地域": "2低層",
        "第1種中高層住居専用地域": "1中高",
        "第2種中高層住居専用地域": "2中高",
        "第1種住居地域": "1住",
        "第2種住居地域": "2住",
    }
)

# HTMLツール(ZNAME/JUKYO)が扱える10区分。1低層・2低層・田園住居は非対応。
HTML_SUPPORTED_KEYS = {
    "1中高", "2中高", "1住", "2住", "準住", "近商", "商業", "準工", "工業", "工専",
}

FAR_STEPS = [100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
BCR_STEPS = [30, 40, 50, 60, 70, 80]


def resolve_zone(code: Optional[int], name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """コードと名称から区分キーを決める。

    コード体系は整備年次で変わったことがある（設計書 2.2 実装上の注意）ため、
    両者が食い違う場合は名称を優先する。どちらからも決められない場合は
    None を返し、呼び出し側で手動選択に落とす。

    Returns (zone_key_or_None, warning_message_or_None).
    """
    name_key = NAME_TO_KEY.get(name) if name else None
    code_key = CODE_TO_KEY.get(code) if code is not None else None

    if name_key is not None:
        if code_key is not None and code_key != name_key:
            return name_key, (
                f"コード({code}→{code_key})と名称('{name}'→{name_key})が一致しません。"
                "名称を優先しました。"
            )
        return name_key, None
    if code_key is not None:
        return code_key, f"名称 '{name}' を認識できないため、コード({code})から推定しました。"
    return None, f"用途地域を認識できません（コード={code}, 名称='{name}'）。手動選択してください。"


def prorate(zone_areas: Sequence[dict], total_area: float) -> Tuple[float, float]:
    """容積率・建蔽率を面積按分する（設計書 3.6）。

    按分容積率 = Σ(部分面積 × その部分の容積率) ÷ 敷地全体面積

    zone_areas の各要素は {"far": float, "bcr": float, "area": float}。
    total_area が敷地の一部しかA29でカバーされていない場合、その未カバー
    分は0扱いになる（呼び出し側でカバー率を別途警告すること）。
    """
    if total_area <= 0:
        raise ValueError("total_area must be positive")
    far_sum = sum(z["area"] * z["far"] for z in zone_areas)
    bcr_sum = sum(z["area"] * z["bcr"] for z in zone_areas)
    return far_sum / total_area, bcr_sum / total_area


def boundary_warning(value: float, steps: Sequence[float] = FAR_STEPS, margin_ratio: float = 0.03) -> Optional[str]:
    """按分値が区分の境界値に近いときに警告する（設計書 2.2）。

    境界値付近では、境界精度に起因するわずかなずれが容積率区分（延いては
    道路斜線の適用距離や日影規制の対象/対象外）を反転させうるため。
    margin_ratio はステップ間隔に対する相対許容幅。
    """
    for step in steps:
        if step <= 0:
            continue
        if abs(value - step) / step <= margin_ratio:
            return f"按分値 {value:.1f} が区分の境界値 {step} に近接しています。都市計画図で要確認。"
    return None


def coverage_warning(covered_area: float, site_area: float, min_ratio: float = 0.98) -> Optional[str]:
    """A29で覆われた面積が敷地面積に対して不足している場合の警告。

    国土数値情報の境界精度は数m程度（設計書 2.2「精度の限界」）。
    """
    if site_area <= 0:
        return None
    ratio = covered_area / site_area
    if ratio < min_ratio:
        return (
            f"敷地面積の {ratio * 100:.1f}% しか用途地域データに覆われていません"
            "（境界データの精度限界、または敷地がデータ範囲外の可能性）。"
        )
    return None
