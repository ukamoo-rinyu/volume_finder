"""辺の条件入力テーブル (設計書 2.3, 4.1).

辺 i は敷地ポリゴンの頂点 i -> i+1（1始まり）。既存HTMLツールの
ETYPE と同じ4区分（道路・水面等・公園広場・隣地）を使う。
"""

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)

# key -> (表示名, 幅員入力を有効にするか)。HTML の ETYPE と対応させる。
EDGE_TYPES = [
    ("road", "道路", True),
    ("sui", "水面・線路敷等", True),
    ("koen", "公園・広場", True),
    ("rinchi", "隣地", False),
]
_TYPE_LABELS = {k: label for k, label, _ in EDGE_TYPES}
_TYPE_HAS_WIDTH = {k: has_width for k, _, has_width in EDGE_TYPES}

COL_INDEX, COL_LENGTH, COL_BEARING, COL_TYPE, COL_WIDTH = range(5)


def _bearing_label(deg: float) -> str:
    dirs = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"]
    i = int((deg + 22.5) // 45) % 8
    return f"{dirs[i]} ({deg:.0f}°)"


class EdgeTable(QTableWidget):
    """敷地の辺ごとに「接する状況」「幅員」を編集するテーブル。"""

    edgesChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self.setHorizontalHeaderLabels(["辺", "長さ (m)", "方位", "接する状況", "幅員 (m)"])
        header = self.horizontalHeader()
        # 数値・方位の列は内容に合わせた固定幅にし、余った幅はすべて
        # 「接する状況」に回す（既定の自動幅だとコンボボックスが
        # 見切れるほど狭くなってしまうため）。
        header.setSectionResizeMode(COL_INDEX, QHeaderView.Fixed)
        header.setSectionResizeMode(COL_LENGTH, QHeaderView.Fixed)
        header.setSectionResizeMode(COL_BEARING, QHeaderView.Fixed)
        header.setSectionResizeMode(COL_TYPE, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_WIDTH, QHeaderView.Fixed)
        self.setColumnWidth(COL_INDEX, 32)
        self.setColumnWidth(COL_LENGTH, 64)
        self.setColumnWidth(COL_BEARING, 90)
        self.setColumnWidth(COL_WIDTH, 90)
        self.verticalHeader().setVisible(False)
        self._edges = []  # list of {"type": str, "width": float} aligned to rows

    def set_edges(self, edge_geometries, initial_edges=None):
        """edge_geometries: core.geometry.polygon_edges() の戻り値。

        initial_edges があれば、同じ辺数の場合に type/width を引き継ぐ
        （設計書 2.3「前回値の引き継ぎ」）。
        """
        self.setRowCount(0)
        n = len(edge_geometries)
        carry = (
            initial_edges
            if initial_edges is not None and len(initial_edges) == n
            else [None] * n
        )
        self._edges = []
        self.setRowCount(n)
        for row, eg in enumerate(edge_geometries):
            prev = carry[row] or {}
            etype = prev.get("type", "rinchi")
            width = float(prev.get("width") or 0)
            self._edges.append({"type": etype, "width": width})

            idx_item = QTableWidgetItem(str(eg["index"]))
            idx_item.setFlags(idx_item.flags() & ~Qt.ItemIsEditable)
            self.setItem(row, COL_INDEX, idx_item)

            len_item = QTableWidgetItem(f"{eg['length']:.1f}")
            len_item.setFlags(len_item.flags() & ~Qt.ItemIsEditable)
            self.setItem(row, COL_LENGTH, len_item)

            bear_item = QTableWidgetItem(_bearing_label(eg["bearing"]))
            bear_item.setFlags(bear_item.flags() & ~Qt.ItemIsEditable)
            self.setItem(row, COL_BEARING, bear_item)

            combo = QComboBox()
            for key, label, _ in EDGE_TYPES:
                combo.addItem(label, key)
            combo.setCurrentIndex(max(0, [k for k, _, _ in EDGE_TYPES].index(etype)))
            combo.currentIndexChanged.connect(lambda _, r=row: self._on_type_changed(r))
            self.setCellWidget(row, COL_TYPE, combo)

            spin = QDoubleSpinBox()
            spin.setRange(0, 999)
            spin.setDecimals(1)
            spin.setSuffix(" m")
            spin.setValue(width)
            spin.setEnabled(_TYPE_HAS_WIDTH[etype])
            spin.valueChanged.connect(lambda _, r=row: self._on_width_changed(r))
            self.setCellWidget(row, COL_WIDTH, spin)

        self.edgesChanged.emit()

    def _on_type_changed(self, row):
        combo = self.cellWidget(row, COL_TYPE)
        spin = self.cellWidget(row, COL_WIDTH)
        key = combo.currentData()
        self._edges[row]["type"] = key
        has_width = _TYPE_HAS_WIDTH[key]
        spin.setEnabled(has_width)
        if not has_width:
            spin.setValue(0)
        self.edgesChanged.emit()

    def _on_width_changed(self, row):
        spin = self.cellWidget(row, COL_WIDTH)
        self._edges[row]["width"] = spin.value()
        self.edgesChanged.emit()

    def edges(self):
        """[{"type": "road"|"sui"|"koen"|"rinchi", "width": float}, ...] を辺番号順で返す。"""
        return [dict(e) for e in self._edges]

    def set_edge_type(self, row, key):
        """接する状況の初期推定を反映する（設計書 2.3-2. 幅員は空欄のまま）。"""
        combo = self.cellWidget(row, COL_TYPE)
        if combo is None or key not in _TYPE_LABELS:
            return
        combo.setCurrentIndex([k for k, _, _ in EDGE_TYPES].index(key))
