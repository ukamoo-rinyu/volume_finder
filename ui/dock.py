"""ドックウィジェット本体 (設計書 4.1)。段階1〜3の範囲:

  敷地選択 → 辺の一覧生成 → 幅員などの手入力
  → A29から用途地域を取得・按分
  → 最大ボリューム探索（単一用途地域・単純な直方体1棟のみ）
  → 結果をレイヤ／JSONで書き出し

用途地域・容積率・建蔽率は、A29から自動取得した結果を初期値として
使うが、常に手動で上書きできる（コード/名称が読めない・複数区分が
複雑な場合の逃げ道として。設計書 2.2「一致しない場合は警告を出して
手動選択に落とす」）。JSONの `zone` はHTMLツールが理解できる10区分の
いずれかである必要があるため、選択肢もその10区分に限定している。

設計書2.1.1の処理順は「union → 最大部分抽出 → 穴除去 → 簡素化 →
EPSG:6674へ再投影」だが、簡素化の許容誤差(m)や面積変化率(%)は
メートル単位のCRSでないと意味を持たないため、本実装では
「EPSG:6674へ再投影」を最初に行い、以降の処理をすべて投影後の
メートル座標で行う。処理内容そのものは設計書と同じ。

A29レイヤは、あらかじめQGISプロジェクトに読み込まれている前提とする
（設計書10章の未決事項#2。ファイル配置・エンコーディング判定はQGIS側の
レイヤ読み込みに委ね、本プラグインでは扱わない）。
"""

import os

from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeatureRequest,
    QgsGeometry,
    QgsMapLayerProxyModel,
    QgsPointXY,
    QgsProject,
)
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core import geometry as geo
from ..core import regulation as reg
from ..core import search as srch
from ..core import shadow as sh
from ..core import zoning
from ..io import html_export
from ..io import json_export as jx
from ..io import layer_export
from .edge_table import EdgeTable
from .result_table import ResultTable
from .search_task import SearchTask

# HTMLツール (ZNAME/JUKYO) が理解できる10区分のみ。順序は既存HTMLのselectに合わせる。
ZONE_LABELS = [
    ("1中高", "第1種中高層住居専用地域"),
    ("2中高", "第2種中高層住居専用地域"),
    ("1住", "第1種住居地域"),
    ("2住", "第2種住居地域"),
    ("準住", "準住居地域"),
    ("近商", "近隣商業地域"),
    ("商業", "商業地域"),
    ("準工", "準工業地域"),
    ("工業", "工業地域"),
    ("工専", "工業専用地域"),
]
HTML_ZONE_KEYS = {k for k, _ in ZONE_LABELS}

# 建蔽率の緩和 (HTMLの#bcrx と同一の4択)。
BCR_RELAX_LABELS = [
    ("0", "なし"),
    ("10", "角地 ＋10%"),
    ("10f", "防火地域＋耐火 ＋10%"),
    ("20", "角地＋防火耐火 ＋20%"),
]

TARGET_CRS = "EPSG:6674"
AREA_WARN_RATIO = 0.005  # 0.5% (設計書 2.1.1)
SETBACK_MIN_ROAD_WIDTH = 4.0
SETBACK_MIN_FRONTAGE = 2.0

# 計算精度プリセット（設計書4.1「計算精度」）。標準を既定にする。
# aspect_ratios/shrink_factorsは精度によらず固定し、日影の時間刻みと
# 格子解像度の目安(max_grid_span)だけを変える。
PRECISION_PRESETS = {
    "粗い（速い）": {"shadow_dt_minutes": 15.0, "max_grid_span": 80.0},
    "標準": {"shadow_dt_minutes": 5.0, "max_grid_span": 150.0},
    "精密（遅い）": {"shadow_dt_minutes": 2.0, "max_grid_span": 250.0},
}
ROTATION_STEP_CHOICES = [5.0, 10.0, 15.0, 30.0, 45.0]


def _largest_polygon_ring(geom):
    """QgsGeometryから最大面積パーツの外周（core.geometry互換の点リスト）を取り出す。

    穴あき・マルチパートでも、探索のzone_regions用途（重心が領域内に
    入るかの判定）には外周だけで十分なので、穴・他パーツは無視する
    （設計書2.1.1で敷地そのものに使った簡略化と同じ考え方）。
    """
    if geom is None or geom.isEmpty():
        return None
    if geom.isMultipart():
        multi = geom.asMultiPolygon()
        if not multi:
            return None
        areas = [QgsGeometry.fromPolygonXY([part[0]]).area() for part in multi]
        best = areas.index(max(areas))
        exterior = multi[best][0]
    else:
        poly = geom.asPolygon()
        if not poly:
            return None
        exterior = poly[0]
    return [(p.x(), p.y()) for p in exterior]


class VolumeFinderDock(QDockWidget):
    def __init__(self, iface, parent=None):
        super().__init__("建築規制ボリューム検討", parent)
        self.iface = iface
        self.setObjectName("VolumeFinderDock")

        self._site_local = None  # CCW正規化済み・重心原点のローカルXY [(x,y),...]
        self._site_local_6674 = None  # 同じ並びのEPSG:6674座標（辺長計算用）
        self._origin_latlng = None  # (lat, lng)
        self._flipped = False
        self._site_warnings = []
        self._source_layer_name = None
        self._source_fids = []
        self._last_edges = None  # 前回値の引き継ぎ用 (設計書 2.3-3)

        self._zone_results = []  # [{"zone_key","name","far","bcr","area"}, ...]
        self._zone_warnings = []
        self._far_prorated = None
        self._bcr_prorated = None
        self._shadow_strictest = None  # (zone_key, far, rule) or None
        self._shadow_target_zones = []  # 設計書3.5: 影の到達範囲にある対象区域 [{"zone_key","name","far"}, ...]

        self._search_task = None
        self._search_results = []  # List[core.search.Candidate]

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        site_box = QGroupBox("① 敷地")
        site_form = QFormLayout(site_box)
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        site_form.addRow("敷地レイヤ", self.layer_combo)
        self.use_selected_btn = QPushButton("選択中の地物を使う")
        self.use_selected_btn.clicked.connect(self.build_site_from_selection)
        site_form.addRow(self.use_selected_btn)
        self.area_label = QLabel("―")
        site_form.addRow("敷地面積", self.area_label)
        self.site_warning_label = QLabel("")
        self.site_warning_label.setWordWrap(True)
        self.site_warning_label.setStyleSheet("color:#B23A2B;")
        site_form.addRow(self.site_warning_label)
        layout.addWidget(site_box)

        zone_box = QGroupBox("② 用途地域")
        zone_layout = QVBoxLayout(zone_box)
        zone_fetch_form = QFormLayout()
        self.a29_layer_combo = QgsMapLayerComboBox()
        self.a29_layer_combo.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        zone_fetch_form.addRow("A29レイヤ", self.a29_layer_combo)
        zone_layout.addLayout(zone_fetch_form)
        self.fetch_zone_btn = QPushButton("用途地域を取得")
        self.fetch_zone_btn.clicked.connect(self.fetch_zoning)
        zone_layout.addWidget(self.fetch_zone_btn)
        self.zone_result_label = QLabel("")
        self.zone_result_label.setWordWrap(True)
        zone_layout.addWidget(self.zone_result_label)
        self.zone_warning_label = QLabel("")
        self.zone_warning_label.setWordWrap(True)
        self.zone_warning_label.setStyleSheet("color:#B23A2B;")
        zone_layout.addWidget(self.zone_warning_label)
        accuracy_note = QLabel("参考値。正確には都市計画図で確認してください。")
        accuracy_note.setWordWrap(True)
        accuracy_note.setStyleSheet("color:#666;font-size:11px;")
        zone_layout.addWidget(accuracy_note)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        zone_layout.addWidget(sep)

        zone_manual_form = QFormLayout()
        self.zone_combo = QComboBox()
        for key, label in ZONE_LABELS:
            self.zone_combo.addItem(label, key)
        self.zone_combo.setCurrentIndex([k for k, _ in ZONE_LABELS].index("1住"))
        zone_manual_form.addRow("用途地域（JSON出力用・手動確認/上書き可）", self.zone_combo)
        self.far_spin = QSpinBox()
        self.far_spin.setRange(50, 1300)
        self.far_spin.setSingleStep(50)
        self.far_spin.setValue(200)
        self.far_spin.setSuffix(" %")
        zone_manual_form.addRow("按分容積率", self.far_spin)
        self.bcr_spin = QSpinBox()
        self.bcr_spin.setRange(30, 100)
        self.bcr_spin.setSingleStep(10)
        self.bcr_spin.setValue(60)
        self.bcr_spin.setSuffix(" %")
        zone_manual_form.addRow("按分建蔽率", self.bcr_spin)
        self.bcr_relax_combo = QComboBox()
        for key, label in BCR_RELAX_LABELS:
            self.bcr_relax_combo.addItem(label, key)
        zone_manual_form.addRow("建蔽率の緩和", self.bcr_relax_combo)
        zone_layout.addLayout(zone_manual_form)
        layout.addWidget(zone_box)

        edge_box = QGroupBox("③ 境界線の条件")
        edge_layout = QVBoxLayout(edge_box)
        self.edge_table = EdgeTable()
        self.edge_table.edgesChanged.connect(self._update_setback_check)
        edge_layout.addWidget(self.edge_table)
        self.setback_label = QLabel("")
        edge_layout.addWidget(self.setback_label)
        layout.addWidget(edge_box)

        adv_box = QGroupBox("④ 敷地整形の設定")
        adv_form = QFormLayout(adv_box)
        self.simplify_spin = QDoubleSpinBox()
        self.simplify_spin.setRange(0, 5)
        self.simplify_spin.setDecimals(2)
        self.simplify_spin.setSingleStep(0.05)
        self.simplify_spin.setValue(0.1)
        self.simplify_spin.setSuffix(" m")
        adv_form.addRow("簡素化の許容誤差", self.simplify_spin)
        layout.addWidget(adv_box)

        search_box = QGroupBox("⑤ 探索条件")
        search_form = QFormLayout()
        self.min_setback_spin = QDoubleSpinBox()
        self.min_setback_spin.setRange(0, 20)
        self.min_setback_spin.setDecimals(1)
        self.min_setback_spin.setValue(1.0)
        self.min_setback_spin.setSuffix(" m")
        search_form.addRow("最小離隔", self.min_setback_spin)
        self.floor_height_spin = QDoubleSpinBox()
        self.floor_height_spin.setRange(2.0, 6.0)
        self.floor_height_spin.setDecimals(1)
        self.floor_height_spin.setValue(3.5)
        self.floor_height_spin.setSuffix(" m")
        search_form.addRow("階高", self.floor_height_spin)
        self.max_floors_spin = QSpinBox()
        self.max_floors_spin.setRange(1, 60)
        self.max_floors_spin.setValue(20)
        search_form.addRow("階数上限", self.max_floors_spin)
        self.rotation_step_combo = QComboBox()
        for step in ROTATION_STEP_CHOICES:
            self.rotation_step_combo.addItem(f"{step:.0f}°", step)
        self.rotation_step_combo.setCurrentIndex(ROTATION_STEP_CHOICES.index(15.0))
        search_form.addRow("回転角の刻み", self.rotation_step_combo)
        self.precision_combo = QComboBox()
        for label in PRECISION_PRESETS:
            self.precision_combo.addItem(label)
        self.precision_combo.setCurrentText("標準")
        search_form.addRow("計算精度", self.precision_combo)
        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(1, 20)
        self.max_results_spin.setValue(3)
        search_form.addRow("表示件数", self.max_results_spin)
        search_box_layout = QVBoxLayout(search_box)
        search_box_layout.addLayout(search_form)

        btn_row = QVBoxLayout()
        self.search_run_btn = QPushButton("探索を実行")
        self.search_run_btn.clicked.connect(self.run_search)
        self.search_run_btn.setEnabled(False)
        btn_row.addWidget(self.search_run_btn)
        self.search_cancel_btn = QPushButton("中断")
        self.search_cancel_btn.clicked.connect(self.cancel_search)
        self.search_cancel_btn.setEnabled(False)
        btn_row.addWidget(self.search_cancel_btn)
        search_box_layout.addLayout(btn_row)

        self.search_progress_bar = QProgressBar()
        self.search_progress_bar.setValue(0)
        search_box_layout.addWidget(self.search_progress_bar)
        self.search_progress_label = QLabel("")
        search_box_layout.addWidget(self.search_progress_label)
        self.search_best_label = QLabel("")
        search_box_layout.addWidget(self.search_best_label)
        layout.addWidget(search_box)

        result_box = QGroupBox("⑥ 結果")
        result_layout = QVBoxLayout(result_box)
        self.result_table = ResultTable()
        result_layout.addWidget(self.result_table)
        result_btn_row = QVBoxLayout()
        self.export_layer_btn = QPushButton("レイヤに出力")
        self.export_layer_btn.clicked.connect(self.export_results_layer)
        self.export_layer_btn.setEnabled(False)
        result_btn_row.addWidget(self.export_layer_btn)
        result_layout.addLayout(result_btn_row)
        layout.addWidget(result_box)

        export_box = QGroupBox("⑦ 書き出し")
        export_layout = QVBoxLayout(export_box)
        self.export_btn = QPushButton("JSONで書き出し（HTMLツール用）")
        self.export_btn.clicked.connect(self.export_json)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        self.export_html_check = QCheckBox("HTMLファイルとしても書き出す（開くだけで確認できます）")
        self.export_html_check.setChecked(True)
        export_layout.addWidget(self.export_html_check)
        self.export_msg = QLabel("")
        self.export_msg.setWordWrap(True)
        export_layout.addWidget(self.export_msg)
        layout.addWidget(export_box)

        layout.addStretch(1)
        root.setLayout(layout)

        # ドックを画面に対して縦に短くフロート表示したときでも全項目に
        # たどり着けるよう、スクロールエリアに包む（項目数が多いため）。
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        self.setWidget(scroll)

    # ------------------------------------------------------------------
    # 敷地の取得・整形 (設計書 2.1.1)
    # ------------------------------------------------------------------
    def build_site_from_selection(self):
        layer = self.layer_combo.currentLayer()
        if layer is None:
            QMessageBox.warning(self, "敷地", "敷地レイヤを選択してください。")
            return
        feats = list(layer.selectedFeatures())
        if not feats:
            QMessageBox.warning(self, "敷地", "レイヤ上で地物を選択してください（複数選択可）。")
            return

        warnings = []
        target_crs = QgsCoordinateReferenceSystem(TARGET_CRS)
        to_target = QgsCoordinateTransform(layer.crs(), target_crs, QgsProject.instance())

        geoms = []
        for f in feats:
            g = QgsGeometry(f.geometry())
            if layer.crs() != target_crs:
                g.transform(to_target)
            geoms.append(g)

        combined = QgsGeometry.unaryUnion(geoms) if len(geoms) > 1 else geoms[0]
        if combined is None or combined.isEmpty():
            QMessageBox.warning(self, "敷地", "結合後のジオメトリが空になりました。選択した地物を確認してください。")
            return

        if combined.isMultipart():
            multi = combined.asMultiPolygon()
            areas = [QgsGeometry.fromPolygonXY([part[0]]).area() for part in multi]
            best = areas.index(max(areas))
            if len(multi) > 1:
                warnings.append(f"マルチポリゴンのため最大面積の部分のみを使用しました（他 {len(multi) - 1} 個を除外）。")
            polygon_rings = multi[best]
        else:
            polygon_rings = combined.asPolygon()

        if len(polygon_rings) > 1:
            warnings.append(f"穴あきポリゴンのため外周のみを使用しました（穴 {len(polygon_rings) - 1} 個を無視）。")
        exterior = polygon_rings[0]

        pts_before = geo.remove_consecutive_duplicates([(p.x(), p.y()) for p in exterior])
        if len(pts_before) < 3:
            QMessageBox.warning(self, "敷地", "有効なポリゴンになりませんでした。")
            return

        tol = self.simplify_spin.value()
        pts_simplified = geo.simplify_polygon(pts_before, tolerance=tol) if tol > 0 else pts_before
        ratio = geo.area_change_ratio(pts_before, pts_simplified)
        if ratio >= AREA_WARN_RATIO:
            warnings.append(f"簡素化により面積が {ratio * 100:.2f}% 変化しました。許容誤差を見直してください。")

        pts_ccw, flipped = geo.to_ccw(pts_simplified)

        self._site_local_6674 = pts_ccw
        self._flipped = flipped
        self._site_warnings = warnings
        self._source_layer_name = layer.name()
        self._source_fids = [f.id() for f in feats]

        area_m2 = geo.polygon_area(pts_ccw)
        self.area_label.setText(f"{area_m2:,.1f} m²")
        self.site_warning_label.setText("\n".join(warnings))

        cx, cy = geo.polygon_centroid(pts_ccw)
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        to_wgs84 = QgsCoordinateTransform(target_crs, wgs84, QgsProject.instance())
        origin_pt = to_wgs84.transform(QgsPointXY(cx, cy))
        origin_lat, origin_lng = origin_pt.y(), origin_pt.x()
        self._origin_latlng = (origin_lat, origin_lng)

        site_local = []
        for x, y in pts_ccw:
            ll_pt = to_wgs84.transform(QgsPointXY(x, y))
            site_local.append(geo.latlng_to_local(ll_pt.y(), ll_pt.x(), origin_lat, origin_lng))
        self._site_local = site_local

        edge_geoms = geo.polygon_edges(pts_ccw)
        self.edge_table.set_edges(edge_geoms, initial_edges=self._last_edges)
        self._last_edges = None

        self.export_btn.setEnabled(True)
        self.search_run_btn.setEnabled(True)
        self._update_setback_check()

    # ------------------------------------------------------------------
    # 用途地域の取得・按分 (設計書 2.2, 3.6)
    # ------------------------------------------------------------------
    def fetch_zoning(self):
        if not self._site_local_6674:
            QMessageBox.warning(self, "用途地域", "先に敷地を取得してください。")
            return
        a29_layer = self.a29_layer_combo.currentLayer()
        if a29_layer is None:
            QMessageBox.warning(self, "用途地域", "A29レイヤを選択してください。")
            return

        target_crs = QgsCoordinateReferenceSystem(TARGET_CRS)
        site_geom = QgsGeometry.fromPolygonXY(
            [[QgsPointXY(x, y) for x, y in self._site_local_6674]]
        )
        site_area = site_geom.area()

        # 敷地周辺で空間フィルタをかけてから処理する（設計書 2.2:
        # A29は都道府県単位で数十MBになりうるため）。
        to_a29_crs = QgsCoordinateTransform(target_crs, a29_layer.crs(), QgsProject.instance())
        site_geom_in_a29_crs = QgsGeometry(site_geom)
        if a29_layer.crs() != target_crs:
            site_geom_in_a29_crs.transform(to_a29_crs)
        request = QgsFeatureRequest().setFilterRect(site_geom_in_a29_crs.boundingBox())

        to_target = QgsCoordinateTransform(a29_layer.crs(), target_crs, QgsProject.instance())
        results = []
        warnings = []
        missing_fields = False
        for f in a29_layer.getFeatures(request):
            try:
                code_raw = f["A29_004"]
                name = f["A29_005"]
                bcr = f["A29_006"]
                far = f["A29_007"]
            except KeyError:
                missing_fields = True
                break
            g = QgsGeometry(f.geometry())
            if a29_layer.crs() != target_crs:
                g.transform(to_target)
            inter = site_geom.intersection(g)
            if inter is None or inter.isEmpty():
                continue
            area = inter.area()
            if area < 0.01:
                continue
            code = int(code_raw) if code_raw is not None else None
            zone_key, resolve_warning = zoning.resolve_zone(code, name)
            if resolve_warning:
                warnings.append(f"「{name}」（コード{code}）: {resolve_warning}")
            results.append(
                {
                    "zone_key": zone_key,
                    "name": name,
                    "far": float(far) if far is not None else 0.0,
                    "bcr": float(bcr) if bcr is not None else 0.0,
                    "area": area,
                    "polygon": _largest_polygon_ring(inter),  # 段階5: 探索の zone_regions 用
                }
            )

        if missing_fields:
            QMessageBox.critical(
                self,
                "用途地域",
                "A29レイヤに A29_004/A29_005/A29_006/A29_007 の属性が見つかりません。"
                "国土数値情報A29のレイヤであることを確認してください。",
            )
            return
        if not results:
            QMessageBox.warning(self, "用途地域", "敷地と重なる用途地域が見つかりませんでした。")
            return

        covered_area = sum(r["area"] for r in results)
        cov_warning = zoning.coverage_warning(covered_area, site_area)
        if cov_warning:
            warnings.append(cov_warning)

        far_prorated, bcr_prorated = zoning.prorate(results, site_area)
        far_boundary_warning = zoning.boundary_warning(far_prorated, zoning.FAR_STEPS)
        if far_boundary_warning:
            warnings.append(far_boundary_warning)

        shadow_candidates = [(r["zone_key"], r["far"]) for r in results if r["zone_key"]]

        # 設計書3.5「影が落ちる先の用途地域」（法56条の2第4項）。
        # 敷地自身が日影規制の対象区域外でも、高さ10mを超える建築物が
        # 近くの対象区域に影を落とす場合は、影が落ちる側の用途地域の
        # 基準が適用される。HTMLツールでは周辺のデータを持てず断念して
        # いた項目（GISだからこそ実現できる）。
        target_zones = []
        max_h = self.max_floors_spin.value() * self.floor_height_spin.value()
        reach = sh.shadow_reach_distance(max_h)
        if reach > 0:
            buffered = site_geom.buffer(reach, 8)
            buffered_in_a29_crs = QgsGeometry(buffered)
            if a29_layer.crs() != target_crs:
                buffered_in_a29_crs.transform(to_a29_crs)
            breq = QgsFeatureRequest().setFilterRect(buffered_in_a29_crs.boundingBox())
            seen = {(r["zone_key"], r["far"]) for r in results}
            for f in a29_layer.getFeatures(breq):
                try:
                    code_raw = f["A29_004"]
                    name = f["A29_005"]
                    far_v = f["A29_007"]
                except KeyError:
                    continue
                g = QgsGeometry(f.geometry())
                if a29_layer.crs() != target_crs:
                    g.transform(to_target)
                if not g.intersects(buffered):
                    continue
                code = int(code_raw) if code_raw is not None else None
                zkey, _w = zoning.resolve_zone(code, name)
                far_num = float(far_v) if far_v is not None else 0.0
                if not zkey or (zkey, far_num) in seen:
                    continue
                seen.add((zkey, far_num))
                if reg.hikage_rule(zkey, far_num):
                    distance = site_geom.distance(g)
                    target_zones.append({"zone_key": zkey, "name": name, "far": far_num, "distance": distance})

        self._shadow_target_zones = target_zones
        all_shadow_candidates = shadow_candidates + [(z["zone_key"], z["far"]) for z in target_zones]
        strictest = reg.strictest_hikage(all_shadow_candidates)

        self._zone_results = results
        self._zone_warnings = warnings
        self._far_prorated = far_prorated
        self._bcr_prorated = bcr_prorated
        self._shadow_strictest = strictest

        lines = []
        for r in sorted(results, key=lambda r: -r["area"]):
            label = reg.ZNAME.get(r["zone_key"], r["name"])
            lines.append(f"{label}　{r['far']:.0f}%　{r['bcr']:.0f}%　{r['area']:,.1f}m²")
        lines.append(f"按分容積率 {far_prorated:.1f}%　按分建蔽率 {bcr_prorated:.1f}%")
        distinct_regulated = {(z, f) for z, f in all_shadow_candidates if reg.hikage_rule(z, f)}
        site_own_regulated = {(z, f) for z, f in shadow_candidates if reg.hikage_rule(z, f)}
        if strictest:
            zone_key, far, rule = strictest
            extra = f"（他{len(distinct_regulated) - 1}区分あり、厳しい方を採用）" if len(distinct_regulated) > 1 else ""
            from_target = (zone_key, far) not in site_own_regulated
            source_note = "影が落ちる先の区域（法56条の2第4項）" if from_target else "敷地自身の区域"
            lines.append(
                f"日影規制：{reg.ZNAME.get(zone_key, zone_key)}(容積率{far:.0f}%)の基準"
                f" mh={rule['mh']}m t1={rule['t1']}分 t2={rule['t2']}分 を採用{extra}　[{source_note}]"
            )
            if target_zones:
                names = "、".join(sorted({reg.ZNAME.get(z["zone_key"], z["name"]) for z in target_zones}))
                lines.append(f"（影の到達範囲 約{reach:.0f}m以内に対象区域あり：{names}）")
        elif target_zones:
            # ここには来ないはず（target_zonesはhikage_rule判定済みのみ格納）だが念のため
            lines.append("日影規制：周辺に対象区域がありますが基準を決定できませんでした。")
        else:
            lines.append(
                "日影規制：敷地・周辺（影の到達範囲内）とも対象区域が見つかりません"
                "（大阪市の6区分外、または未判定の区分あり）。"
            )
        self.zone_result_label.setText("\n".join(lines))
        self.zone_warning_label.setText("\n".join(warnings))

        # 手動確認欄に初期値として反映（常に上書き可能）。
        self.far_spin.setValue(round(far_prorated))
        self.bcr_spin.setValue(round(bcr_prorated))
        largest_html_zone = next(
            (r["zone_key"] for r in sorted(results, key=lambda r: -r["area"]) if r["zone_key"] in HTML_ZONE_KEYS),
            None,
        )
        if largest_html_zone:
            self.zone_combo.setCurrentIndex([k for k, _ in ZONE_LABELS].index(largest_html_zone))

    # ------------------------------------------------------------------
    # 最大ボリューム探索 (設計書 3章、4.2の応答性確保)
    # ------------------------------------------------------------------
    def run_search(self):
        if not self._site_local_6674:
            QMessageBox.warning(self, "探索", "先に敷地を取得してください。")
            return
        edges = self.edge_table.edges()
        if not any(e["type"] == "road" and e["width"] > 0 for e in edges):
            QMessageBox.warning(
                self, "探索",
                "道路に接する辺の幅員が入力されていません。前面道路幅員による容積率制限が"
                "0%として扱われるため、このままでは候補が見つかりません。",
            )
            return

        preset = PRECISION_PRESETS[self.precision_combo.currentText()]
        shadow_override = self._shadow_strictest[2] if self._shadow_strictest else None

        # 設計書3.6「複数用途地域」: 用途地域・斜線は按分できないため、A29と
        # 敷地の交差領域を候補ごとの重心判定に使う（core.search._local_zone）。
        # 面積の大きい順に並べ、重なりがあれば大きい方を優先する。
        zone_regions = [
            {"zone_key": r["zone_key"], "far": r["far"], "polygon": r["polygon"]}
            for r in sorted(self._zone_results, key=lambda r: -r["area"])
            if r["zone_key"] and r["polygon"]
        ]

        # 最小離隔ぶん内側にオフセットした建築可能範囲は、GEOSの堅牢な
        # buffer()で計算してcore.searchに渡す。core.geometry.offset_polygon_per_edge
        # （辺ごとのミター結合による簡易実装）は、切れ込みのある凹んだ
        # 敷地で正しくオフセットできず、指定した最小離隔より実際には
        # 敷地境界に近い場所に建物が置かれてしまうことがあるため。
        min_setback = self.min_setback_spin.value()
        site_geom = QgsGeometry.fromPolygonXY(
            [[QgsPointXY(x, y) for x, y in self._site_local_6674]]
        )
        buildable_geom = site_geom.buffer(-min_setback, 8) if min_setback > 0 else site_geom
        buildable = _largest_polygon_ring(buildable_geom)
        if not buildable:
            QMessageBox.warning(
                self, "探索",
                f"最小離隔 {min_setback:.1f}m を確保すると敷地内に建築可能な範囲が残りません。"
                "最小離隔を小さくしてください。",
            )
            return

        params = srch.SearchParams(
            min_setback=min_setback,
            buildable_override=buildable,
            floor_height=self.floor_height_spin.value(),
            max_floors=self.max_floors_spin.value(),
            rotation_step_deg=float(self.rotation_step_combo.currentData()),
            shadow_dt_minutes=preset["shadow_dt_minutes"],
            shadow_cell=None,
            bcr_relax=self.bcr_relax_combo.currentData(),
            max_results=self.max_results_spin.value(),
            hikage_override=shadow_override,
            zone_regions=zone_regions or None,
        )
        zone_key = self.zone_combo.currentData()
        far = self.far_spin.value()
        bcr = self.bcr_spin.value()

        self._search_task = SearchTask(
            "建築規制ボリューム探索", list(self._site_local_6674), edges, zone_key, far, bcr, params
        )
        self._search_task.progress_update.connect(self._on_search_progress)
        self._search_task.taskCompleted.connect(self._on_search_finished)
        self._search_task.taskTerminated.connect(self._on_search_finished)
        QgsApplication.taskManager().addTask(self._search_task)

        self.search_run_btn.setEnabled(False)
        self.search_cancel_btn.setEnabled(True)
        self.search_progress_bar.setValue(0)
        self.search_progress_label.setText("候補を生成中…")
        self.search_best_label.setText("")

    def cancel_search(self):
        if self._search_task is not None:
            self._search_task.cancel()

    def _on_search_progress(self, done, total, best):
        self.search_progress_bar.setMaximum(max(1, total))
        self.search_progress_bar.setValue(done)
        self.search_progress_label.setText(f"候補 {done}/{total} を評価中")
        if best is not None:
            self.search_best_label.setText(f"現在の最良：{best.floors}階 延床 {best.floor_area:,.0f}m²")

    def _on_search_finished(self, *_args):
        task = self._search_task
        if task is None:
            return
        self._search_task = None  # taskCompleted/taskTerminatedが両方来ても二重処理しない
        self.search_run_btn.setEnabled(True)
        self.search_cancel_btn.setEnabled(False)

        if task.exception is not None:
            QMessageBox.critical(self, "探索", f"探索中にエラーが発生しました：{task.exception}")
            return

        self._search_results = task.results
        self.result_table.set_results(self._search_results)
        self.export_layer_btn.setEnabled(bool(self._search_results))
        if self._search_results:
            best = self._search_results[0]
            self.search_progress_label.setText(f"完了：候補 {len(self._search_results)} 件")
            self.search_best_label.setText(f"最良案：{best.floors}階 延床 {best.floor_area:,.0f}m²")
        else:
            self.search_progress_label.setText("完了：条件を満たす候補が見つかりませんでした")

    def export_results_layer(self):
        if not self._search_results:
            QMessageBox.warning(self, "レイヤ出力", "先に探索を実行してください。")
            return
        layer_export.add_candidates_to_project(self._search_results, crs_authid=TARGET_CRS)

    # ------------------------------------------------------------------
    # 接道義務チェック (設計書 2.3-4)
    # ------------------------------------------------------------------
    def _update_setback_check(self):
        if not self._site_local_6674:
            self.setback_label.setText("")
            return
        edges = self.edge_table.edges()
        edge_geoms = geo.polygon_edges(self._site_local_6674)
        ok_edges = [
            eg for e, eg in zip(edges, edge_geoms)
            if e["type"] == "road" and e["width"] >= SETBACK_MIN_ROAD_WIDTH
            and eg["length"] >= SETBACK_MIN_FRONTAGE
        ]
        if ok_edges:
            best = max(ok_edges, key=lambda eg: eg["length"])
            self.setback_label.setStyleSheet("color:#2A7A3B;")
            self.setback_label.setText(
                f"接道義務：OK（辺{best['index']}、{best['length']:.1f}mが幅員{SETBACK_MIN_ROAD_WIDTH:.0f}m以上の道路に接する）"
            )
        else:
            self.setback_label.setStyleSheet("color:#B23A2B;")
            self.setback_label.setText(
                f"接道義務：未確認（幅員{SETBACK_MIN_ROAD_WIDTH:.0f}m以上の道路に{SETBACK_MIN_FRONTAGE:.0f}m以上接する辺が必要）"
            )

    # ------------------------------------------------------------------
    # 探索結果 -> HTMLツールのbuilding形式への変換 (設計書 1.3, 8.2)
    # ------------------------------------------------------------------
    def _selected_candidate(self):
        """結果テーブルで選択中の案。未選択なら1位（先頭）を既定にする。"""
        if not self._search_results:
            return None
        row = self.result_table.selected_row()
        if row is not None and 0 <= row < len(self._search_results):
            return self._search_results[row]
        return self._search_results[0]

    def _candidate_to_building(self, candidate):
        """core.search.Candidate（EPSG:6674座標）をHTMLツールのbuilding形式に変換する。

        cx/cyは敷地の頂点と同じ変換（EPSG:6674 -> 緯度経度 -> 重心原点の
        ローカルXY、設計書2.4）を通す必要がある。単純にEPSG:6674の値を
        そのまま使うと、敷地ポリゴン（既にローカルXY変換済み）とズレる。
        rotはEPSG:6674とローカルXYの東西南北がほぼ一致する
        （どちらも赤道方向基準の平面直角座標系相当）ため変換せず流用する。
        """
        target_crs = QgsCoordinateReferenceSystem(TARGET_CRS)
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        to_wgs84 = QgsCoordinateTransform(target_crs, wgs84, QgsProject.instance())
        ll_pt = to_wgs84.transform(QgsPointXY(candidate.cx, candidate.cy))
        lx, ly = geo.latlng_to_local(ll_pt.y(), ll_pt.x(), self._origin_latlng[0], self._origin_latlng[1])
        return {
            "name": f"候補{candidate.rank}",
            "W": round(candidate.W, 2),
            "D": round(candidate.D, 2),
            "rot": round(candidate.rot, 2),
            "cx": round(lx, 3),
            "cy": round(ly, 3),
            "fh": self.floor_height_spin.value(),
            "nf": candidate.floors,
            "ph": 0,
        }

    # ------------------------------------------------------------------
    # JSON書き出し (設計書 5.2)
    # ------------------------------------------------------------------
    def export_json(self):
        if self._site_local is None or self._origin_latlng is None:
            QMessageBox.warning(self, "書き出し", "先に敷地を取得してください。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "JSONで書き出し", "", "JSON Files (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"

        edges = self.edge_table.edges()
        zone_key = self.zone_combo.currentData()
        candidate = self._selected_candidate()
        buildings = [self._candidate_to_building(candidate)] if candidate is not None else []
        qgis_extra = {
            "flipped": self._flipped,
            "source_layer": self._source_layer_name,
            "source_fids": self._source_fids,
            "simplify_tolerance_m": self.simplify_spin.value(),
            "warnings": self._site_warnings,
        }
        if self._zone_results:
            qgis_extra["zones"] = [
                {
                    "name": reg.ZNAME.get(r["zone_key"], r["name"]),
                    "far": r["far"],
                    "bcr": r["bcr"],
                    "area": r["area"],
                }
                for r in self._zone_results
            ]
            qgis_extra["far_prorated"] = self._far_prorated
            qgis_extra["bcr_prorated"] = self._bcr_prorated
            qgis_extra["zone_warnings"] = self._zone_warnings
            if self._shadow_strictest:
                sk, sfar, srule = self._shadow_strictest
                # 敷地自身の区域(distance=0) + 影が落ちる先の対象区域(設計書3.5)を
                # 1つのリストにまとめる（設計書5.2の shadow_zones スキーマに合わせる）。
                own_zone_pairs = {(r["zone_key"], r["far"]) for r in self._zone_results if r["zone_key"]}
                shadow_zones = [
                    {"name": reg.ZNAME.get(z, z), "far": f, "mh": rule["mh"], "t1": rule["t1"], "t2": rule["t2"], "distance": 0.0}
                    for z, f in own_zone_pairs
                    if (rule := reg.hikage_rule(z, f))
                ]
                shadow_zones.extend(
                    {
                        "name": reg.ZNAME.get(z["zone_key"], z["name"]),
                        "far": z["far"],
                        "mh": reg.hikage_rule(z["zone_key"], z["far"])["mh"],
                        "t1": reg.hikage_rule(z["zone_key"], z["far"])["t1"],
                        "t2": reg.hikage_rule(z["zone_key"], z["far"])["t2"],
                        "distance": round(z["distance"], 1),
                    }
                    for z in self._shadow_target_zones
                )
                qgis_extra["shadow_zones"] = shadow_zones
                qgis_extra["shadow_applied"] = {"name": reg.ZNAME.get(sk, sk), "far": sfar, **srule}
        if candidate is not None:
            qgis_extra["rank"] = candidate.rank
            qgis_extra["binding"] = candidate.binding
        doc = jx.build_json(
            self._site_local,
            self._origin_latlng[0],
            self._origin_latlng[1],
            edges,
            zone=zone_key,
            far=self.far_spin.value(),
            bcr=self.bcr_spin.value(),
            bcrx=self.bcr_relax_combo.currentData(),
            buildings=buildings,
            qgis_extra=qgis_extra,
        )
        self._last_edges = edges  # 次回の敷地取得時に引き継ぐ
        try:
            jx.write_json(path, doc)
        except OSError as e:
            QMessageBox.critical(self, "書き出し", f"書き出しに失敗しました：{e}")
            return

        html_path = None
        if self.export_html_check.isChecked():
            html_path = os.path.splitext(path)[0] + ".html"
            try:
                html_export.write_standalone_html(html_path, doc)
            except OSError as e:
                QMessageBox.warning(self, "書き出し", f"JSONは書き出せましたが、HTMLの書き出しに失敗しました：{e}")
                html_path = None

        candidate_note = f"（{candidate.rank}位の案を含む）" if candidate is not None else "（建物なし。先に探索を実行すると案も書き出せます）"
        if html_path:
            self.export_msg.setText(f"書き出しました：{path}\n{html_path}{candidate_note}")
        else:
            self.export_msg.setText(f"書き出しました：{path}{candidate_note}")
