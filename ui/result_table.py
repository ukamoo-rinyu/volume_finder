"""結果一覧テーブル (ver0.2.0仕様3.7の多目的スコアを表示に反映)。"""

from qgis.PyQt.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

COLUMNS = [
    "順位", "スコア", "アンカー", "階数", "高さ(m)", "建築面積(m²)", "延床面積(m²)",
    "容積率(%)", "建蔽率(%)", "有効空地(m²)", "南面採光", "効いている制約",
]


class ResultTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, len(COLUMNS), parent)
        self.setHorizontalHeaderLabels(COLUMNS)
        self.horizontalHeader().setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)

    def set_results(self, candidates):
        self.setRowCount(len(candidates))
        for row, c in enumerate(candidates):
            values = [
                str(c.rank),
                f"{c.score:.2f}",
                c.anchor,
                f"{c.floors}階",
                f"{c.height:.1f}",
                f"{c.footprint_area:,.1f}",
                f"{c.floor_area:,.1f}",
                f"{c.far_pct:.1f}",
                f"{c.bcr_pct:.1f}",
                f"{c.open_area:,.0f}（短辺{c.open_min_side:.1f}m）",
                f"{c.light_ratio:.2f}",
                "・".join(c.binding),
            ]
            for col, v in enumerate(values):
                self.setItem(row, col, QTableWidgetItem(v))

    def selected_row(self):
        rows = self.selectionModel().selectedRows() if self.selectionModel() else []
        return rows[0].row() if rows else None
