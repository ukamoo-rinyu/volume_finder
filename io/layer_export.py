"""候補建物レイヤの出力 (設計書 5.1)。

段階3では「候補建物」レイヤのみを出力する。「建築可能範囲」「等時間
日影線」「後退距離の測線」は将来の段階で追加する（可視化の充実は
探索アルゴリズム本体より優先度を下げた）。
"""

from qgis.core import (
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant

from ..core import geometry as geo

FIELDS = [
    ("rank", QVariant.Int),
    ("floors", QVariant.Int),
    ("height_m", QVariant.Double),
    ("footprint_m2", QVariant.Double),
    ("floor_area_m2", QVariant.Double),
    ("far_pct", QVariant.Double),
    ("bcr_pct", QVariant.Double),
    ("binding", QVariant.String),
    ("verdict", QVariant.String),
]


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
                c.floors,
                round(c.height, 2),
                round(c.footprint_area, 2),
                round(c.floor_area, 2),
                round(c.far_pct, 2),
                round(c.bcr_pct, 2),
                "・".join(c.binding),
                "適合",  # search() が返す候補はすべて制約を満たしたもののみ
            ]
        )
        feats.append(feat)
    pr.addFeatures(feats)
    layer.updateExtents()
    return layer


def add_candidates_to_project(candidates, crs_authid: str = "EPSG:6674", layer_name: str = "候補建物") -> QgsVectorLayer:
    layer = candidates_to_layer(candidates, crs_authid, layer_name)
    QgsProject.instance().addMapLayer(layer)
    return layer
