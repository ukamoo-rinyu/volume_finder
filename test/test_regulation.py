"""core/regulation.py の基準値を検証する。

設計書 8.1: 「core/regulation.py の値は、既存HTMLツールと同じ結果に
なることを単体テストで確認する」。ここでは hikage-osaka-v4_23.html の
hikageRule(z,f) 関数の分岐をそのまま突き合わせる。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from volume_finder.core import regulation


def test_hikage_rule_matches_html_source_values():
    # HTML: hikageRule() の4分岐をそのまま照合
    assert regulation.hikage_rule("1中高", 200) == {"mh": 4, "t1": 4, "t2": 2.5}
    assert regulation.hikage_rule("2中高", 300) == {"mh": 4, "t1": 5, "t2": 3}
    for zone in ("1住", "2住", "準住"):
        assert regulation.hikage_rule(zone, 200) == {"mh": 4, "t1": 5, "t2": 3}
    assert regulation.hikage_rule("準工", 200) == {"mh": 6.5, "t1": 5, "t2": 3}


def test_hikage_rule_none_outside_the_six_categories():
    # HTMLでも対象外 (大阪市の6区分のみ対象、設計書 8.1)
    assert regulation.hikage_rule("近商", 400) is None
    assert regulation.hikage_rule("商業", 400) is None
    assert regulation.hikage_rule("1中高", 300) is None  # 容積率が違えば対象外
    assert regulation.hikage_rule("2中高", 200) is None  # 同上


def test_jukyo_and_zname_match_html():
    assert regulation.JUKYO == ["1中高", "2中高", "1住", "2住", "準住"]
    assert set(regulation.ZNAME) == {
        "1中高", "2中高", "1住", "2住", "準住", "近商", "商業", "準工", "工業", "工専",
    }


def test_strictest_hikage_picks_smallest_t2():
    candidates = [("1住", 200), ("準工", 200), ("1中高", 200)]
    zone, far, rule = regulation.strictest_hikage(candidates)
    assert zone == "1中高"
    assert far == 200
    assert rule["t2"] == 2.5


def test_strictest_hikage_ignores_non_regulated_zones():
    candidates = [("近商", 400), ("商業", 500)]
    assert regulation.strictest_hikage(candidates) is None


def test_strictest_hikage_single_candidate():
    candidates = [("2住", 200)]
    zone, far, rule = regulation.strictest_hikage(candidates)
    assert zone == "2住"


def test_road_rule_matches_html_source():
    # 住居系: k=1.25, L は far 区分で 20/25/30/35
    assert regulation.road_rule("1住", 200) == {"k": 1.25, "L": 20, "kind": "住居系"}
    assert regulation.road_rule("2住", 250) == {"k": 1.25, "L": 25, "kind": "住居系"}
    assert regulation.road_rule("準住", 350) == {"k": 1.25, "L": 30, "kind": "住居系"}
    assert regulation.road_rule("1中高", 500) == {"k": 1.25, "L": 35, "kind": "住居系"}
    # 商業系: k=1.5, L は far 区分で 20..50
    assert regulation.road_rule("商業", 400) == {"k": 1.5, "L": 20, "kind": "商業系"}
    assert regulation.road_rule("近商", 1300) == {"k": 1.5, "L": 50, "kind": "商業系"}
    # 工業系（その他）: k=1.5, L は住居系と同じ区分
    assert regulation.road_rule("工業", 200) == {"k": 1.5, "L": 20, "kind": "工業系"}
    assert regulation.road_rule("工専", 500) == {"k": 1.5, "L": 35, "kind": "工業系"}


def test_rin_rule_matches_html_source():
    assert regulation.rin_rule("1住") == {"H0": 20, "k": 1.25}
    assert regulation.rin_rule("商業") == {"H0": 31, "k": 2.5}


def test_far_coef_matches_html_source():
    assert regulation.far_coef("1住") == 0.4
    assert regulation.far_coef("商業") == 0.6


def test_bcr_limit_matches_html_source():
    assert regulation.bcr_limit(60, "0") == 60
    assert regulation.bcr_limit(60, "10") == 70
    assert regulation.bcr_limit(60, "10f") == 70
    assert regulation.bcr_limit(60, "20") == 80
    assert regulation.bcr_limit(80, "10f") == float("inf")


def test_hikage_relax_matches_html_source():
    assert regulation.hikage_relax("rinchi", 0) == 0.0
    assert regulation.hikage_relax("road", 0) == 0.0
    assert regulation.hikage_relax("road", 8) == 4.0  # <=10: width/2
    assert regulation.hikage_relax("road", 16) == 11.0  # >10: width-5
    assert regulation.hikage_relax("sui", 6) == 3.0
    assert regulation.hikage_relax("koen", 6) == 3.0
