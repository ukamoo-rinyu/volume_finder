"""候補建物レイヤの出力 (設計書 5.1)。

段階3では「候補建物」レイヤのみを出力する。「建築可能範囲」「等時間
日影線」「後退距離の測線」は将来の段階で追加する（可視化の充実は
探索アルゴリズム本体より優先度を下げた）。

順位(rank)ごとにカテゴリ分けしたシンボロジーを既定で適用する。候補の
矩形は重なり合うことが多く、単色のままだと地図上でどれがどの順位か
見分けがつかないため。
"""

import colorsys

from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsFeature,
    QgsField,
    QgsFillSymbol,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRendererCategory,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor

from ..core import geometry as geo

FIELDS = [
    ("rank", QVariant.Int),
    ("score", QVariant.Double),
    ("anchor", QVariant.String),
    ("floors", QVariant.Int),
    ("height_m", QVariant.Double),
    ("footprint_m2", QVariant.Double),
    ("floor_area_m2", QVariant.Double),
    ("far_pct", QVariant.Double),
    ("bcr_pct", QVariant.Double),
    ("open_area_m2", QVariant.Double),
    ("open_min_side_m", QVariant.Double),
    ("light_ratio", QVariant.Double),
    ("binding", QVariant.String),
    ("verdict", QVariant.String),
]


def _rank_renderer(ranks) -> QgsCategorizedSymbolRenderer:
    """rank（順位）フィールドでカテゴリ分けしたシンボロジーを作る。

    色は色相をランクの昇順に均等に割り振るだけの自前実装（QgsStyleの
    名前付きカラーランプはQGISのバージョン・環境によって用意されている
    ものが違うため、依存しない）。ranks は昇順（1位から）を想定。
    """
    categories = []
    n = len(ranks)
    for i, rank in enumerate(ranks):
        hue = (i / n) if n > 1 else 0.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.92)
        color = QColor(int(r * 255), int(g * 255), int(b * 255))
        symbol = QgsFillSymbol.createSimple(
            {"color": color.name(), "outline_color": "#333333", "outline_width": "0.4", "style": "solid"}
        )
        symbol.setOpacity(0.65)
        categories.append(QgsRendererCategory(rank, symbol, f"{rank}位"))
    return QgsCategorizedSymbolRenderer("rank", categories)


def candidates_to_layer(candidates, crs_authid: str = "EPSG:6674", layer_name: str = "候補建物") -> QgsVectorLayer:
    layer = QgsVectorLayer(f"Polygon?crs={crs_authid}", layer_name, "memory")
    pr = layer.dataProvider()
    pr.addAttributes([QgsField(name, qtype) for name, qtype in FIELDS])
    layer.updateFields()

    feats = []
    for c in candidates:
        corners = geo.rectangle_corners(c.cx, c.cy, c.W, c.D, c.rot)
        feat = QgsFeature(layer.fields())
        feat.setGeometry(QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in corners]]))
        feat.setAttributes(
            [
                c.rank,
                round(c.score, 3),
                c.anchor,
                c.floors,
                round(c.height, 2),
                round(c.footprint_area, 2),
                round(c.floor_area, 2),
                round(c.far_pct, 2),
                round(c.bcr_pct, 2),
                round(c.open_area, 2),
                round(c.open_min_side, 2),
                round(c.light_ratio, 3),
                "・".join(c.binding),
                "適合",  # search() が返す候補はすべて制約を満たしたもののみ
            ]
        )
        feats.append(feat)
    pr.addFeatures(feats)
    layer.updateExtents()

    ranks = sorted({c.rank for c in candidates})
    if ranks:
        layer.setRenderer(_rank_renderer(ranks))
    return layer


def add_candidates_to_project(candidates, crs_authid: str = "EPSG:6674", layer_name: str = "候補建物") -> QgsVectorLayer:
    layer = candidates_to_layer(candidates, crs_authid, layer_name)
    QgsProject.instance().addMapLayer(layer)
    layer.triggerRepaint()
    return layer


# --- 複数棟配置 (ver0.3.0、分棟・L型、core.search.MultiCandidate) ---

MULTI_FIELDS = [
    ("rank", QVariant.Int),
    ("building", QVariant.String),  # "A"/"B"
    ("score", QVariant.Double),
    ("floors", QVariant.Int),
    ("height_m", QVariant.Double),
    ("footprint_m2", QVariant.Double),
    ("floor_area_m2", QVariant.Double),
    ("total_far_pct", QVariant.Double),  # 2棟合計の容積率(%)
    ("total_bcr_pct", QVariant.Double),  # 2棟合計の建蔽率(%)
    ("binding", QVariant.String),
    ("verdict", QVariant.String),
]


def multi_candidates_to_layer(
    candidates, crs_authid: str = "EPSG:6674", layer_name: str = "候補建物（複数棟）"
) -> QgsVectorLayer:
    """MultiCandidate（2棟ぶんのBuildingPlacementを持つ、ver0.3.0）を、
    棟ごとに1フィーチャずつのレイヤに書き出す。rankフィールドは2棟で
    共通の値になる（同じ案の中の2棟だと分かるように）ので、既存の
    _rank_renderer をそのまま再利用してカテゴリ分けできる。
    """
    layer = QgsVectorLayer(f"Polygon?crs={crs_authid}", layer_name, "memory")
    pr = layer.dataProvider()
    pr.addAttributes([QgsField(name, qtype) for name, qtype in MULTI_FIELDS])
    layer.updateFields()

    feats = []
    for c in candidates:
        for b in c.buildings:
            corners = geo.rectangle_corners(b.cx, b.cy, b.W, b.D, b.rot)
            feat = QgsFeature(layer.fields())
            feat.setGeometry(QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in corners]]))
            feat.setAttributes(
                [
                    c.rank,
                    b.label,
                    round(c.score, 3),
                    b.floors,
                    round(b.height, 2),
                    round(b.footprint_area, 2),
                    round(b.floor_area, 2),
                    round(c.far_pct, 2),
                    round(c.bcr_pct, 2),
                    "・".join(c.binding),
                    "適合",
                ]
            )
            feats.append(feat)
    pr.addFeatures(feats)
    layer.updateExtents()

    ranks = sorted({c.rank for c in candidates})
    if ranks:
        layer.setRenderer(_rank_renderer(ranks))
    return layer


def add_multi_candidates_to_project(
    candidates, crs_authid: str = "EPSG:6674", layer_name: str = "候補建物（複数棟）"
) -> QgsVectorLayer:
    layer = multi_candidates_to_layer(candidates, crs_authid, layer_name)
    QgsProject.instance().addMapLayer(layer)
    layer.triggerRepaint()
    return layer
