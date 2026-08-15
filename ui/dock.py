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

import hashlib
import json
import math
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
    QgsSettings,
)
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
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
from .result_table import MultiResultTable, ResultTable
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

# 境界線の条件（辺ごとの接する状況・幅員）を敷地の形状ごとに永続化する
# QgsSettings のキー接頭辞。QGISを再起動しても残る（レジストリ／iniファイル
# にQGIS本体が書き込む場所へ保存されるため）。敷地の識別はレイヤ・地物IDでは
# なく形状の指紋（_site_fingerprint）で行うので、別レイヤから選び直したり
# 手描きし直したりしても、同じ形であれば前回の入力を思い出せる。
EDGE_MEMORY_SETTINGS_GROUP = "volume_finder/edge_memory"
EDGE_MEMORY_ROUND_M = 0.1  # 指紋計算時の座標丸め(m)。この程度のクリック誤差は同一視する
SETBACK_MIN_ROAD_WIDTH = 4.0
SETBACK_MIN_FRONTAGE = 2.0

# 計算精度プリセット（設計書4.1「計算精度」）。標準を既定にする。
# 平面形の候補生成（size_fractions/anchors）は精度によらず固定し、
# 日影の時間刻みと格子解像度の目安(max_grid_span)だけを変える。
PRECISION_PRESETS = {
    "粗い（速い）": {"shadow_dt_minutes": 15.0, "max_grid_span": 80.0},
    "標準": {"shadow_dt_minutes": 5.0, "max_grid_span": 150.0},
    "精密（遅い）": {"shadow_dt_minutes": 2.0, "max_grid_span": 250.0},
}
ROTATION_STEP_CHOICES = [5.0, 10.0, 15.0, 30.0, 45.0]

# 回転角モード (ver0.2.0仕様1章)。既定は「境界線平行」。
ROTATION_MODE_CHOICES = [
    ("edge", "境界線平行（既定）"),
    ("orthogonal", "直交のみ（主要接道辺）"),
    ("free", "自由（比較用）"),
]

# アンカー配置 (ver0.2.0仕様2章)。既定はすべてON。
ANCHOR_CHOICES = [(k, srch.ANCHOR_LABELS[k]) for k in srch.ANCHOR_KEYS]

# 施設タイプ (ver0.3.0)。指定なし(None)＝従来どおり制約なしの自由探索。
FACILITY_TYPE_CHOICES = [(None, "指定なし（自由）")] + [
    (key, ft.name) for key, ft in srch.FACILITY_TYPES.items()
]


def _facility_type_summary(ftype):
    """施設タイプの主な数値を短い説明文にする（UIの確認用ラベル）。"""
    if ftype is None:
        return "プロポーション・階数範囲などの制約なしで探索します。"
    parts = [
        f"階高{ftype.floor_height:.1f}m",
        f"階数{ftype.min_floors}〜{ftype.max_floors}階" if ftype.max_floors else f"階数{ftype.min_floors}階以上",
    ]
    if ftype.min_footprint_dim:
        parts.append(f"最低短辺{ftype.min_footprint_dim:.0f}m")
    if ftype.max_aspect_ratio:
        parts.append(f"長辺/短辺{ftype.max_aspect_ratio:.1f}以下")
    if ftype.min_footprint_area:
        parts.append(f"最低建築面積{ftype.min_footprint_area:.0f}m²")
    if ftype.requires_yard:
        parts.append(f"隣接空地{ftype.yard_min_area:.0f}m²以上（短辺{ftype.yard_min_side:.0f}m以上）必須")
    return "　".join(parts)


# 評価の重み (ver0.2.0仕様3.7章) のスライダー既定値。
WEIGHT_DEFAULTS = {"far": 50, "open": 30, "light": 5, "impact": 0}
WEIGHT_LABELS = {"far": "容積消化率", "open": "有効空地", "light": "南面採光", "impact": "隣地影響"}


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
        self._last_edges = None  # 前回値の引き継ぎ用 (設計書 2.3-3、同一セッション内でのみ有効)
        self._current_edge_fingerprint = None  # QGIS再起動をまたいだ記憶のキー。敷地未取得ならNone

        self._zone_results = []  # [{"zone_key","name","far","bcr","area"}, ...]
        self._zone_warnings = []
        self._far_prorated = None
        self._bcr_prorated = None
        self._shadow_strictest = None  # (zone_key, far, rule) or None。JSON書き出し(HTML表示)用の代表値
        self._shadow_target_zones = []  # 設計書3.5: 影の到達範囲にある対象区域 [{"zone_key","name","far","polygon"}, ...]
        self._hikage_regions = []  # 探索用: [{"mask","rule","name"}, ...]（core.search.SearchParams.hikage_regions）

        self._site_area = 0.0

        self._search_task = None
        self._search_results = []  # List[core.search.Candidate]（類似解間引き後の表示用）
        self._search_pool = []  # List[core.search.Candidate]（正規化済み・重み変更時の再ランキング用）

        # ver0.3.0: 複数棟配置（分棟・L型）。単棟探索とは独立した並行の
        # 結果セット（core.search.search_two_buildings、MultiCandidate）。
        self._multi_search_task = None
        self._multi_search_results = []
        self._multi_search_pool = []

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
        self.zone_combo.currentIndexChanged.connect(lambda _=None: self._invalidate_search_results("用途地域"))
        zone_manual_form.addRow("用途地域（JSON出力用・手動確認/上書き可）", self.zone_combo)
        self.far_spin = QSpinBox()
        self.far_spin.setRange(50, 1300)
        self.far_spin.setSingleStep(50)
        self.far_spin.setValue(200)
        self.far_spin.setSuffix(" %")
        self.far_spin.valueChanged.connect(lambda _=None: self._invalidate_search_results("按分容積率"))
        zone_manual_form.addRow("按分容積率", self.far_spin)
        self.bcr_spin = QSpinBox()
        self.bcr_spin.setRange(30, 100)
        self.bcr_spin.setSingleStep(10)
        self.bcr_spin.setValue(60)
        self.bcr_spin.setSuffix(" %")
        self.bcr_spin.valueChanged.connect(lambda _=None: self._invalidate_search_results("按分建蔽率"))
        zone_manual_form.addRow("按分建蔽率", self.bcr_spin)
        self.bcr_relax_combo = QComboBox()
        for key, label in BCR_RELAX_LABELS:
            self.bcr_relax_combo.addItem(label, key)
        self.bcr_relax_combo.currentIndexChanged.connect(lambda _=None: self._invalidate_search_results("建蔽率の緩和"))
        zone_manual_form.addRow("建蔽率の緩和", self.bcr_relax_combo)
        zone_layout.addLayout(zone_manual_form)
        layout.addWidget(zone_box)

        edge_box = QGroupBox("③ 境界線の条件")
        edge_layout = QVBoxLayout(edge_box)
        self.edge_table = EdgeTable()
        self.edge_table.edgesChanged.connect(self._update_setback_check)
        self.edge_table.edgesChanged.connect(self._persist_current_edges)
        self.edge_table.edgesChanged.connect(lambda: self._invalidate_search_results("境界線の条件"))
        edge_layout.addWidget(self.edge_table)
        self.edge_memory_label = QLabel("")
        self.edge_memory_label.setStyleSheet("color:#666;font-size:11px;")
        self.edge_memory_label.setWordWrap(True)
        edge_layout.addWidget(self.edge_memory_label)
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
        self.facility_type_combo = QComboBox()
        for key, label in FACILITY_TYPE_CHOICES:
            self.facility_type_combo.addItem(label, key)
        search_form.addRow("施設タイプ（任意）", self.facility_type_combo)
        self.facility_type_note = QLabel(_facility_type_summary(None))
        self.facility_type_note.setWordWrap(True)
        self.facility_type_note.setStyleSheet("color:#666;font-size:11px;")
        self.facility_type_combo.currentIndexChanged.connect(self._on_facility_type_changed)
        search_form.addRow(self.facility_type_note)
        self.rotation_mode_combo = QComboBox()
        for key, label in ROTATION_MODE_CHOICES:
            self.rotation_mode_combo.addItem(label, key)
        search_form.addRow("回転角モード", self.rotation_mode_combo)
        self.rotation_step_combo = QComboBox()
        for step in ROTATION_STEP_CHOICES:
            self.rotation_step_combo.addItem(f"{step:.0f}°", step)
        self.rotation_step_combo.setCurrentIndex(ROTATION_STEP_CHOICES.index(15.0))
        search_form.addRow("回転角の刻み（「自由」のときのみ）", self.rotation_step_combo)
        self.precision_combo = QComboBox()
        for label in PRECISION_PRESETS:
            self.precision_combo.addItem(label)
        self.precision_combo.setCurrentText("標準")
        search_form.addRow("計算精度", self.precision_combo)
        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(1, 20)
        self.max_results_spin.setValue(8)
        self.max_results_spin.valueChanged.connect(self._on_weight_changed)
        search_form.addRow("表示件数", self.max_results_spin)
        self.dedupe_threshold_spin = QDoubleSpinBox()
        self.dedupe_threshold_spin.setRange(0, 100)
        self.dedupe_threshold_spin.setDecimals(1)
        self.dedupe_threshold_spin.setValue(10.0)
        self.dedupe_threshold_spin.setSuffix(" m")
        self.dedupe_threshold_spin.valueChanged.connect(self._on_weight_changed)
        search_form.addRow("類似解の間引き距離", self.dedupe_threshold_spin)
        search_box_layout = QVBoxLayout(search_box)
        search_box_layout.addLayout(search_form)

        anchor_label = QLabel("配置するアンカー（寄せ方向、複数選択可）")
        anchor_label.setWordWrap(True)
        search_box_layout.addWidget(anchor_label)
        anchor_grid = QGridLayout()
        self.anchor_checks = {}
        for i, (key, label) in enumerate(ANCHOR_CHOICES):
            cb = QCheckBox(label)
            cb.setChecked(True)
            self.anchor_checks[key] = cb
            anchor_grid.addWidget(cb, i // 3, i % 3)
        search_box_layout.addLayout(anchor_grid)

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

        weight_box = QGroupBox("⑥ 評価の重み（探索し直さず並び替えのみ）")
        weight_form = QFormLayout(weight_box)
        weight_note = QLabel("スライダーを動かすと、直前の探索結果の中で並び替えるだけなので、再探索なしで即座に反映されます。")
        weight_note.setWordWrap(True)
        weight_form.addRow(weight_note)
        self.weight_sliders = {}
        self.weight_value_labels = {}
        for key in ("far", "open", "light", "impact"):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(WEIGHT_DEFAULTS[key])
            slider.valueChanged.connect(self._on_weight_changed)
            value_label = QLabel(str(WEIGHT_DEFAULTS[key]))
            value_label.setFixedWidth(28)
            slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(str(v)))
            hrow = QHBoxLayout()
            hrow.addWidget(slider)
            hrow.addWidget(value_label)
            self.weight_sliders[key] = slider
            self.weight_value_labels[key] = value_label
            weight_form.addRow(WEIGHT_LABELS[key], hrow)
        impact_note = QLabel("「隣地影響」は計算コストが高いため、探索前に0より大きくしておいた場合のみ実測されます。探索後にこのスライダーだけ動かしても差は出ません。")
        impact_note.setWordWrap(True)
        impact_note.setStyleSheet("color:#666;font-size:11px;")
        weight_form.addRow(impact_note)
        layout.addWidget(weight_box)

        result_box = QGroupBox("⑦ 結果")
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

        export_box = QGroupBox("⑧ 書き出し")
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

        layout.addWidget(self._build_multi_box())

        layout.addStretch(1)
        root.setLayout(layout)

        # ドックを画面に対して縦に短くフロート表示したときでも全項目に
        # たどり着けるよう、スクロールエリアに包む（項目数が多いため）。
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        self.setWidget(scroll)

    def _build_multi_box(self):
        """複数棟配置（分棟・L型、ver0.3.0、試験的機能）のUI。

        単棟探索（①〜⑧）とは独立した並行のワークフローとして最後に
        追加する。敷地・用途地域・境界線の条件・探索条件（最小離隔・階高・
        階数上限・回転角モード・計算精度・アンカー）・評価の重みは、
        上のセクションで入力したものをそのまま使い回す（この機能専用の
        入力は「棟間の離隔距離」のみ）。
        """
        box = QGroupBox("複数棟配置を試す（分棟・L型、試験的機能）")
        box_layout = QVBoxLayout(box)

        note = QLabel(
            "敷地を2つのゾーンに分けて、それぞれに1棟ずつ配置する組合せを探索します。"
            "離隔距離を0にすると隙間なく接する配置（L型）になります。"
            "回転は2棟で共有し、階数は棟ごとに独立して二分探索で決めますが、"
            "同時に最適な組合せそのものを厳密に尽くすわけではない近似です。"
            "敷地・用途地域・境界線の条件・探索条件・評価の重みは上の設定を"
            "そのまま使います。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#666;font-size:11px;")
        box_layout.addWidget(note)

        multi_form = QFormLayout()
        self.building_gap_spin = QDoubleSpinBox()
        self.building_gap_spin.setRange(0, 30)
        self.building_gap_spin.setDecimals(1)
        self.building_gap_spin.setValue(3.0)
        self.building_gap_spin.setSuffix(" m")
        multi_form.addRow("棟間の離隔距離（0mでL型）", self.building_gap_spin)
        self.building_type_a_combo = QComboBox()
        for key, label in FACILITY_TYPE_CHOICES:
            self.building_type_a_combo.addItem(label, key)
        multi_form.addRow("A棟タイプ（任意）", self.building_type_a_combo)
        self.building_type_b_combo = QComboBox()
        for key, label in FACILITY_TYPE_CHOICES:
            self.building_type_b_combo.addItem(label, key)
        multi_form.addRow("B棟タイプ（任意）", self.building_type_b_combo)
        box_layout.addLayout(multi_form)

        self.building_type_note = QLabel(f"A棟: {_facility_type_summary(None)}\nB棟: {_facility_type_summary(None)}")
        self.building_type_note.setWordWrap(True)
        self.building_type_note.setStyleSheet("color:#666;font-size:11px;")
        self.building_type_a_combo.currentIndexChanged.connect(self._on_building_type_changed)
        self.building_type_b_combo.currentIndexChanged.connect(self._on_building_type_changed)
        box_layout.addWidget(self.building_type_note)

        multi_btn_row = QVBoxLayout()
        self.multi_search_run_btn = QPushButton("複数棟配置を探索")
        self.multi_search_run_btn.clicked.connect(self.run_multi_search)
        self.multi_search_run_btn.setEnabled(False)
        multi_btn_row.addWidget(self.multi_search_run_btn)
        self.multi_search_cancel_btn = QPushButton("中断")
        self.multi_search_cancel_btn.clicked.connect(self.cancel_multi_search)
        self.multi_search_cancel_btn.setEnabled(False)
        multi_btn_row.addWidget(self.multi_search_cancel_btn)
        box_layout.addLayout(multi_btn_row)

        self.multi_search_progress_bar = QProgressBar()
        self.multi_search_progress_bar.setValue(0)
        box_layout.addWidget(self.multi_search_progress_bar)
        self.multi_search_progress_label = QLabel("")
        box_layout.addWidget(self.multi_search_progress_label)
        self.multi_search_best_label = QLabel("")
        box_layout.addWidget(self.multi_search_best_label)

        self.multi_result_table = MultiResultTable()
        box_layout.addWidget(self.multi_result_table)

        self.export_multi_layer_btn = QPushButton("レイヤに出力（複数棟）")
        self.export_multi_layer_btn.clicked.connect(self.export_multi_results_layer)
        self.export_multi_layer_btn.setEnabled(False)
        box_layout.addWidget(self.export_multi_layer_btn)

        self.export_multi_btn = QPushButton("JSONで書き出し（複数棟、HTMLツール用）")
        self.export_multi_btn.clicked.connect(self.export_multi_json)
        self.export_multi_btn.setEnabled(False)
        box_layout.addWidget(self.export_multi_btn)
        self.export_multi_html_check = QCheckBox("HTMLファイルとしても書き出す（開くだけで確認できます）")
        self.export_multi_html_check.setChecked(True)
        box_layout.addWidget(self.export_multi_html_check)
        self.export_multi_msg = QLabel("")
        self.export_multi_msg.setWordWrap(True)
        box_layout.addWidget(self.export_multi_msg)

        return box

    # ------------------------------------------------------------------
    # 境界線の条件（辺ごとの接する状況・幅員）をQGIS再起動をまたいで記憶する。
    # 「一度探索を行った敷地について、次回以降も選択状況を覚えている」ため、
    # レイヤ・地物IDではなく敷地の形状（EPSG:6674座標）そのものを指紋にする
    # （別レイヤから選び直しても、手描きし直しても同じ形なら思い出せる）。
    # ------------------------------------------------------------------
    @staticmethod
    def _site_fingerprint(pts_6674):
        """敷地形状（CCW正規化済みEPSG:6674座標）から記憶用のキーを作る。

        座標をEDGE_MEMORY_ROUND_M単位に丸めてからハッシュ化することで、
        クリック位置の数cmのブレを同一の敷地とみなす。頂点の並び順は
        to_ccw()で常に一定になる（設計書2.1.2）ので、同じ敷地なら
        再取得しても同じキーになる。
        """
        if not pts_6674:
            return None
        rounded = [
            (round(x / EDGE_MEMORY_ROUND_M), round(y / EDGE_MEMORY_ROUND_M))
            for x, y in pts_6674
        ]
        # QgsSettingsのキー用の非暗号学的なキャッシュキー（衝突耐性・改ざん耐性は
        # 不要）。usedforsecurity=Falseで、セキュリティスキャナ(Bandit等)による
        # 「弱いハッシュ」警告を抑止する（用途上はMD5等でも構わないが、SHA1が
        # 標準ライブラリで最も手軽なため使用）。
        digest = hashlib.sha1(repr(rounded).encode("utf-8"), usedforsecurity=False).hexdigest()[:20]
        return digest

    def _load_persisted_edges(self, fingerprint):
        if not fingerprint:
            return None
        raw = QgsSettings().value(f"{EDGE_MEMORY_SETTINGS_GROUP}/{fingerprint}", None)
        if not raw:
            return None
        try:
            edges = json.loads(raw)
        except (TypeError, ValueError):
            return None
        if not isinstance(edges, list):
            return None
        return edges

    def _persist_current_edges(self):
        """境界線の条件が変わるたびに、現在の敷地の指紋に紐づけて保存する
        （設計書2.3-3の「前回値の引き継ぎ」をQGISの再起動をまたいで効かせる）。"""
        if not self._current_edge_fingerprint:
            return
        edges = self.edge_table.edges()
        if not edges:
            return
        QgsSettings().setValue(
            f"{EDGE_MEMORY_SETTINGS_GROUP}/{self._current_edge_fingerprint}",
            json.dumps(edges),
        )

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
        self._site_area = area_m2
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

        # QGISを再起動しても、同じ形の敷地なら前回の境界線の条件を思い出す。
        # 記憶が無い初めての敷地では、同一セッション内の前回値(_last_edges)に
        # フォールバックする（設計書2.3-3）。
        self._current_edge_fingerprint = self._site_fingerprint(pts_ccw)
        persisted_edges = self._load_persisted_edges(self._current_edge_fingerprint)
        edge_geoms = geo.polygon_edges(pts_ccw)
        self.edge_table.set_edges(edge_geoms, initial_edges=persisted_edges or self._last_edges)
        self._last_edges = None
        self.edge_memory_label.setText(
            "この敷地は以前入力した接する状況・幅員を記憶しています（自動反映済み）。"
            if persisted_edges else ""
        )
        self._invalidate_search_results()  # 敷地が変わったので、古い敷地に対する探索結果は無効

        self.export_btn.setEnabled(True)
        self.search_run_btn.setEnabled(True)
        self.multi_search_run_btn.setEnabled(True)
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
        # 影の到達範囲の見積もりに使う高さ。探索条件の「階数上限」は探索が
        # 試す上限（既定20階）であって実際に建つ高さの見積もりではないため、
        # そのまま使うと現実離れした広い範囲を拾ってしまう（実例:
        # 9,150m²・容積率300%の敷地で階数上限20階のまま計算すると到達範囲
        # 469mになり、明らかに影が届かない遠方の区域まで拾ってしまって
        # いた）。按分容積率・按分建蔽率から「指定建蔽率いっぱいに建てた
        # ときに容積を使い切る階数」（= far/bcr、建築実務でよく使う目安）を
        # 求め、階数上限とのいずれか小さい方を高さの見積もりに使う。
        # 有効高さ（設計書3.5）は建物高さ－測定面高さ(mh)なので、mhは
        # 影を受ける側の区分がまだ分からないこの時点では最小値
        # （regulation.min_regulated_mh、大阪市6区分でmh=4m）を使い、
        # 到達範囲を狭めすぎない（安全側）ようにする。
        est_floors = math.ceil(far_prorated / bcr_prorated) if bcr_prorated > 0 else self.max_floors_spin.value()
        est_floors = max(1, min(est_floors, self.max_floors_spin.value()))
        est_height = est_floors * self.floor_height_spin.value()
        effective_height = max(0.0, est_height - reg.min_regulated_mh())
        reach = sh.shadow_reach_distance(effective_height)
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
                    target_zones.append({
                        "zone_key": zkey, "name": name, "far": far_num, "distance": distance,
                        # 探索でのマスク用（設計書3.5・ver0.2.0）: この区域の実際の
                        # ポリゴン全体（敷地との交差ではなく、区域そのもの）。
                        # 影がこのポリゴンの中に実際に落ちている部分だけを判定に使う。
                        "polygon": _largest_polygon_ring(g),
                    })

        self._shadow_target_zones = target_zones
        # 探索用: 「影が落ちる先の区域」は、それぞれの実際のポリゴン（マスク）
        # の中でしか判定してはいけない（ver0.2.0で修正した不具合。影が実際
        # には届かない方角にある対象区域まで拾って全周を制約していた）。
        # 敷地自身の区域の基準は、探索側(core.search._floors_within_shadow)が
        # candidateごとのlocal_zone/local_farから自動で判定するので、ここに
        # 含める必要はない。
        self._hikage_regions = [
            {"mask": z["polygon"], "rule": reg.hikage_rule(z["zone_key"], z["far"]), "name": reg.ZNAME.get(z["zone_key"], z["name"])}
            for z in target_zones
            if z.get("polygon")
        ]

        all_shadow_candidates = shadow_candidates + [(z["zone_key"], z["far"]) for z in target_zones]
        strictest = reg.strictest_hikage(all_shadow_candidates)

        self._zone_results = results
        self._zone_warnings = warnings
        self._far_prorated = far_prorated
        self._bcr_prorated = bcr_prorated
        self._shadow_strictest = strictest  # JSON書き出し(HTML表示)用の代表値。探索は_hikage_regionsを使う

        lines = []
        for r in sorted(results, key=lambda r: -r["area"]):
            label = reg.ZNAME.get(r["zone_key"], r["name"])
            lines.append(f"{label}　{r['far']:.0f}%　{r['bcr']:.0f}%　{r['area']:,.1f}m²")
        lines.append(f"按分容積率 {far_prorated:.1f}%　按分建蔽率 {bcr_prorated:.1f}%")

        # 日影規制の内訳は、敷地自身の区域（全周に適用）と、影が落ちる先の
        # 各対象区域（それぞれの実際の範囲内だけに適用）を分けて列挙する。
        # 「厳しい方を採用」と1本にまとめてしまうと、対象区域が敷地の一部
        # 方角にしかない場合でも全方位に規制がかかっているかのように誤解
        # されるため（実際に見つかった不具合）。
        site_own_regulated = sorted({(z, f) for z, f in shadow_candidates if reg.hikage_rule(z, f)})
        if site_own_regulated:
            for zone_key, far in site_own_regulated:
                rule = reg.hikage_rule(zone_key, far)
                lines.append(
                    f"日影規制：{reg.ZNAME.get(zone_key, zone_key)}(容積率{far:.0f}%)の基準"
                    f" mh={rule['mh']}m t1={rule['t1']}分 t2={rule['t2']}分（敷地自身の区域、全周に適用）"
                )
        else:
            lines.append("日影規制：敷地自身の区域は対象外です。")
        if target_zones:
            for z in sorted(target_zones, key=lambda z: z["distance"]):
                rule = reg.hikage_rule(z["zone_key"], z["far"])
                mask_note = "" if z.get("polygon") else "　⚠区域の形状が取得できずマスクなし（全周に適用）"
                lines.append(
                    f"日影規制：{reg.ZNAME.get(z['zone_key'], z['name'])}(容積率{z['far']:.0f}%)の基準"
                    f" mh={rule['mh']}m t1={rule['t1']}分 t2={rule['t2']}分"
                    f"（影が落ちる先の区域、法56条の2第4項、約{z['distance']:.0f}m先、その区域の範囲内だけに適用{mask_note}）"
                )
            lines.append(f"（影の到達範囲 約{reach:.0f}m以内を検索）")
        elif not site_own_regulated:
            lines.append(
                "（影の到達範囲 約{:.0f}m以内にも対象区域が見つかりません。大阪市の6区分外、または未判定の区分あり）".format(reach)
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
    def _on_facility_type_changed(self, _index=None):
        """単棟探索の施設タイプ選択（ver0.3.0）。数値の説明ラベルを更新し、
        探索条件が変わったので直前の探索結果を無効化する。"""
        ftype = srch.FACILITY_TYPES.get(self.facility_type_combo.currentData())
        self.facility_type_note.setText(_facility_type_summary(ftype))
        self._invalidate_search_results("施設タイプ")

    def _gather_common_search_inputs(self):
        """単棟探索(run_search)・複数棟配置探索(run_multi_search)で共通する
        入力（境界線条件・複数用途地域の判定用領域・建築可能範囲・アンカー・
        計算精度プリセット）をまとめて取得する。検証に失敗した場合は
        QMessageBoxで理由を表示してNoneを返す（呼び出し側はNoneならreturnする）。
        """
        if not self._site_local_6674:
            QMessageBox.warning(self, "探索", "先に敷地を取得してください。")
            return None
        edges = self.edge_table.edges()
        if not any(e["type"] == "road" and e["width"] > 0 for e in edges):
            QMessageBox.warning(
                self, "探索",
                "道路に接する辺の幅員が入力されていません。前面道路幅員による容積率制限が"
                "0%として扱われるため、このままでは候補が見つかりません。",
            )
            return None

        preset = PRECISION_PRESETS[self.precision_combo.currentText()]

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
            return None

        anchors = [key for key, cb in self.anchor_checks.items() if cb.isChecked()]
        if not anchors:
            QMessageBox.warning(self, "探索", "アンカーを少なくとも1つ選択してください。")
            return None

        return {
            "edges": edges, "preset": preset, "zone_regions": zone_regions,
            "min_setback": min_setback, "buildable": buildable, "anchors": anchors,
        }

    def _build_search_params(self, common, **overrides):
        """_gather_common_search_inputs()の戻り値からSearchParamsを組み立てる。
        overridesはSearchParamsの追加・上書き用（例: 複数棟探索のbuilding_gap）。
        """
        kwargs = dict(
            min_setback=common["min_setback"],
            buildable_override=common["buildable"],
            floor_height=self.floor_height_spin.value(),
            max_floors=self.max_floors_spin.value(),
            rotation_mode=self.rotation_mode_combo.currentData(),
            rotation_step_deg=float(self.rotation_step_combo.currentData()),
            anchors=common["anchors"],
            shadow_dt_minutes=common["preset"]["shadow_dt_minutes"],
            shadow_cell=None,
            bcr_relax=self.bcr_relax_combo.currentData(),
            max_results=self.max_results_spin.value(),
            dedupe_threshold=self.dedupe_threshold_spin.value(),
            weight_far=self.weight_sliders["far"].value(),
            weight_open=self.weight_sliders["open"].value(),
            weight_light=self.weight_sliders["light"].value(),
            weight_impact=self.weight_sliders["impact"].value(),
            hikage_regions=self._hikage_regions or None,
            zone_regions=common["zone_regions"] or None,
        )
        kwargs.update(overrides)
        return srch.SearchParams(**kwargs)

    def run_search(self):
        common = self._gather_common_search_inputs()
        if common is None:
            return
        facility_type = srch.FACILITY_TYPES.get(self.facility_type_combo.currentData())
        params = self._build_search_params(common, facility_type=facility_type)
        edges = common["edges"]
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
        self._search_pool = task.pool  # 重み変更時に再探索なしで並び替えるためのプール（仕様3.7）
        self.result_table.set_results(self._search_results)
        self.export_layer_btn.setEnabled(bool(self._search_results))
        if self._search_results:
            best = self._search_results[0]
            self.search_progress_label.setText(f"完了：候補 {len(self._search_results)} 件（プール {len(self._search_pool)} 件）")
            self.search_best_label.setText(f"最良案：{best.floors}階 延床 {best.floor_area:,.0f}m²")
        else:
            self.search_progress_label.setText("完了：条件を満たす候補が見つかりませんでした")

    # ------------------------------------------------------------------
    # 複数棟配置探索 (ver0.3.0、分棟・L型)
    # ------------------------------------------------------------------
    def _on_building_type_changed(self, _index=None):
        """複数棟配置のA棟/B棟タイプ選択（ver0.3.0）。数値の説明ラベルを
        更新し、探索条件が変わったので直前の探索結果を無効化する。"""
        a = srch.FACILITY_TYPES.get(self.building_type_a_combo.currentData())
        b = srch.FACILITY_TYPES.get(self.building_type_b_combo.currentData())
        self.building_type_note.setText(f"A棟: {_facility_type_summary(a)}\nB棟: {_facility_type_summary(b)}")
        self._invalidate_search_results("施設タイプ")

    def run_multi_search(self):
        common = self._gather_common_search_inputs()
        if common is None:
            return
        building_types = (
            srch.FACILITY_TYPES.get(self.building_type_a_combo.currentData()),
            srch.FACILITY_TYPES.get(self.building_type_b_combo.currentData()),
        )
        params = self._build_search_params(
            common, building_gap=self.building_gap_spin.value(), building_types=building_types,
        )
        edges = common["edges"]
        zone_key = self.zone_combo.currentData()
        far = self.far_spin.value()
        bcr = self.bcr_spin.value()

        self._multi_search_task = SearchTask(
            "複数棟配置探索", list(self._site_local_6674), edges, zone_key, far, bcr, params,
            search_fn=srch.search_two_buildings,
        )
        self._multi_search_task.progress_update.connect(self._on_multi_search_progress)
        self._multi_search_task.taskCompleted.connect(self._on_multi_search_finished)
        self._multi_search_task.taskTerminated.connect(self._on_multi_search_finished)
        QgsApplication.taskManager().addTask(self._multi_search_task)

        self.multi_search_run_btn.setEnabled(False)
        self.multi_search_cancel_btn.setEnabled(True)
        self.multi_search_progress_bar.setValue(0)
        self.multi_search_progress_label.setText("分割パターンを生成中…")
        self.multi_search_best_label.setText("")

    def cancel_multi_search(self):
        if self._multi_search_task is not None:
            self._multi_search_task.cancel()

    def _on_multi_search_progress(self, done, total, best):
        self.multi_search_progress_bar.setMaximum(max(1, total))
        self.multi_search_progress_bar.setValue(done)
        self.multi_search_progress_label.setText(f"分割パターン {done}/{total} を評価中")
        if best is not None:
            a, b = best.buildings
            self.multi_search_best_label.setText(
                f"現在の最良：A棟{a.floors}階／B棟{b.floors}階　延床合計 {best.floor_area:,.0f}m²"
            )

    def _on_multi_search_finished(self, *_args):
        task = self._multi_search_task
        if task is None:
            return
        self._multi_search_task = None  # taskCompleted/taskTerminatedが両方来ても二重処理しない
        self.multi_search_run_btn.setEnabled(True)
        self.multi_search_cancel_btn.setEnabled(False)

        if task.exception is not None:
            QMessageBox.critical(self, "複数棟配置", f"探索中にエラーが発生しました：{task.exception}")
            return

        self._multi_search_results = task.results
        self._multi_search_pool = task.pool
        self.multi_result_table.set_results(self._multi_search_results)
        self.export_multi_layer_btn.setEnabled(bool(self._multi_search_results))
        self.export_multi_btn.setEnabled(bool(self._multi_search_results))
        if self._multi_search_results:
            best = self._multi_search_results[0]
            a, b = best.buildings
            self.multi_search_progress_label.setText(
                f"完了：候補 {len(self._multi_search_results)} 件（プール {len(self._multi_search_pool)} 件）"
            )
            self.multi_search_best_label.setText(
                f"最良案：A棟{a.floors}階／B棟{b.floors}階　延床合計 {best.floor_area:,.0f}m²"
            )
        else:
            self.multi_search_progress_label.setText("完了：条件を満たす候補が見つかりませんでした")

    def _on_weight_changed(self, *_args):
        """評価の重みスライダーの変更（仕様3.7「重み変更時はソートし直すだけ」）。

        直前の探索で得たプール（self._search_pool、正規化済みの指標を持つ）を
        使って並び替え・類似解間引きをやり直すだけで、core.search.search() は
        呼ばない。プールが無ければ（まだ探索していない）何もしない。
        複数棟配置のプール（self._multi_search_pool）があれば、同じ重み・
        表示件数・類似解間引き距離でそちらも並び替える
        （core.search.MultiCandidateはCandidateと同名のフィールドを持つため、
        core.search.rerank をそのまま使い回せる）。
        """
        weights = {
            "far": self.weight_sliders["far"].value(),
            "open": self.weight_sliders["open"].value(),
            "light": self.weight_sliders["light"].value(),
            "impact": self.weight_sliders["impact"].value(),
        }
        if self._search_pool:
            self._search_results = srch.rerank(
                self._search_pool, weights, self._site_area,
                dedupe_threshold=self.dedupe_threshold_spin.value(),
                max_results=self.max_results_spin.value(),
            )
            self.result_table.set_results(self._search_results)
            self.export_layer_btn.setEnabled(bool(self._search_results))
            if self._search_results:
                best = self._search_results[0]
                self.search_best_label.setText(f"最良案：{best.floors}階 延床 {best.floor_area:,.0f}m²")
        if self._multi_search_pool:
            self._multi_search_results = srch.rerank(
                self._multi_search_pool, weights, self._site_area,
                dedupe_threshold=self.dedupe_threshold_spin.value(),
                max_results=self.max_results_spin.value(),
            )
            self.multi_result_table.set_results(self._multi_search_results)
            self.export_multi_layer_btn.setEnabled(bool(self._multi_search_results))
            self.export_multi_btn.setEnabled(bool(self._multi_search_results))
            if self._multi_search_results:
                best = self._multi_search_results[0]
                a, b = best.buildings
                self.multi_search_best_label.setText(
                    f"最良案：A棟{a.floors}階／B棟{b.floors}階　延床合計 {best.floor_area:,.0f}m²"
                )

    def _invalidate_search_results(self, reason=None):
        """探索条件（用途地域・容積率・建蔽率・境界条件・敷地）が変わったら、
        古い探索結果を保持し続けない（単棟・複数棟の両方）。

        実際に見つかった不具合の再現: 「用途地域を取得」で按分容積率・
        按分建蔽率が更新されても（例: 200%→300%）、それより前に行った
        探索結果は古い値（200%）のまま画面・レイヤ・JSON出力に残り続け、
        現在表示されている按分値と食い違う案を、それと気づかずに書き出して
        しまっていた。重み・表示件数・類似解間引き距離の変更は再探索
        不要なので、ここでは呼ばない（_on_weight_changed / core.search.rerank
        を使う）。
        """
        if self._search_results or self._search_pool:
            self._search_results = []
            self._search_pool = []
            self.result_table.set_results([])
            self.export_layer_btn.setEnabled(False)
            prefix = f"{reason}が変更されたため、" if reason else ""
            self.search_progress_label.setText(f"{prefix}直前の探索結果は無効になりました。再探索してください。")
            self.search_best_label.setText("")
        if self._multi_search_results or self._multi_search_pool:
            self._multi_search_results = []
            self._multi_search_pool = []
            self.multi_result_table.set_results([])
            self.export_multi_layer_btn.setEnabled(False)
            self.export_multi_btn.setEnabled(False)
            prefix = f"{reason}が変更されたため、" if reason else ""
            self.multi_search_progress_label.setText(f"{prefix}直前の探索結果は無効になりました。再探索してください。")
            self.multi_search_best_label.setText("")

    def export_results_layer(self):
        if not self._search_results:
            QMessageBox.warning(self, "レイヤ出力", "先に探索を実行してください。")
            return
        layer_export.add_candidates_to_project(self._search_results, crs_authid=TARGET_CRS)

    def export_multi_results_layer(self):
        if not self._multi_search_results:
            QMessageBox.warning(self, "レイヤ出力", "先に複数棟配置の探索を実行してください。")
            return
        layer_export.add_multi_candidates_to_project(self._multi_search_results, crs_authid=TARGET_CRS)

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

    def _selected_multi_candidate(self):
        """複数棟配置の結果テーブルで選択中の案。未選択なら1位を既定にする。"""
        if not self._multi_search_results:
            return None
        row = self.multi_result_table.selected_row()
        if row is not None and 0 <= row < len(self._multi_search_results):
            return self._multi_search_results[row]
        return self._multi_search_results[0]

    def _local_xy(self, cx, cy):
        """EPSG:6674座標を、敷地の頂点と同じ変換（-> 緯度経度 -> 重心原点の
        ローカルXY、設計書2.4）でHTMLツールのローカルXYに変換する。単純に
        EPSG:6674の値をそのまま使うと、敷地ポリゴン（既にローカルXY変換済み）
        とズレる。"""
        target_crs = QgsCoordinateReferenceSystem(TARGET_CRS)
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        to_wgs84 = QgsCoordinateTransform(target_crs, wgs84, QgsProject.instance())
        ll_pt = to_wgs84.transform(QgsPointXY(cx, cy))
        return geo.latlng_to_local(ll_pt.y(), ll_pt.x(), self._origin_latlng[0], self._origin_latlng[1])

    def _candidate_to_building(self, candidate):
        """core.search.Candidate（EPSG:6674座標）をHTMLツールのbuilding形式に変換する。

        rotはEPSG:6674とローカルXYの東西南北がほぼ一致する（どちらも赤道
        方向基準の平面直角座標系相当）ため変換せず流用する。
        """
        lx, ly = self._local_xy(candidate.cx, candidate.cy)
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

    def _multi_candidate_to_buildings(self, candidate):
        """core.search.MultiCandidate（ver0.3.0、2棟）をHTMLツールのbuilding
        形式のリスト（2件）に変換する。_candidate_to_buildingの複数棟版。"""
        buildings = []
        for b in candidate.buildings:
            lx, ly = self._local_xy(b.cx, b.cy)
            buildings.append({
                "name": f"{candidate.rank}位 {b.label}棟",
                "W": round(b.W, 2),
                "D": round(b.D, 2),
                "rot": round(b.rot, 2),
                "cx": round(lx, 3),
                "cy": round(ly, 3),
                "fh": self.floor_height_spin.value(),
                "nf": b.floors,
                "ph": 0,
            })
        return buildings

    def _zone_shadow_qgis_extra(self):
        """JSON書き出しの_qgis拡張のうち、用途地域・日影規制の内訳
        （単棟・複数棟の書き出しで共通）をまとめて返す。self._zone_resultsが
        空（用途地域を未取得）なら空dict。"""
        if not self._zone_results:
            return {}
        extra = {
            "zones": [
                {
                    "name": reg.ZNAME.get(r["zone_key"], r["name"]),
                    "far": r["far"],
                    "bcr": r["bcr"],
                    "area": r["area"],
                }
                for r in self._zone_results
            ],
            "far_prorated": self._far_prorated,
            "bcr_prorated": self._bcr_prorated,
            "zone_warnings": self._zone_warnings,
        }
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
            extra["shadow_zones"] = shadow_zones
            extra["shadow_applied"] = {
                "name": reg.ZNAME.get(sk, sk), "far": sfar, **srule,
                "from_target": (sk, sfar) not in own_zone_pairs,  # 設計書3.5: 影が落ちる先の区域の基準か
            }
        return extra

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
        qgis_extra.update(self._zone_shadow_qgis_extra())
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

    def export_multi_json(self):
        """複数棟配置（ver0.3.0）の選択中の案を、HTMLツール用JSONで書き出す。
        単棟版のexport_json()と並存する別ボタン（設計書上の使い分けは
        「単棟の結果」と「複数棟の結果」で完全に独立させる、という方針）。
        """
        if self._site_local is None or self._origin_latlng is None:
            QMessageBox.warning(self, "書き出し", "先に敷地を取得してください。")
            return
        candidate = self._selected_multi_candidate()
        if candidate is None:
            QMessageBox.warning(self, "書き出し", "先に複数棟配置の探索を実行してください。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "JSONで書き出し（複数棟）", "", "JSON Files (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"

        edges = self.edge_table.edges()
        zone_key = self.zone_combo.currentData()
        buildings = self._multi_candidate_to_buildings(candidate)
        qgis_extra = {
            "flipped": self._flipped,
            "source_layer": self._source_layer_name,
            "source_fids": self._source_fids,
            "simplify_tolerance_m": self.simplify_spin.value(),
            "warnings": self._site_warnings,
            "rank": candidate.rank,
            "binding": candidate.binding,
        }
        qgis_extra.update(self._zone_shadow_qgis_extra())
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
        self._last_edges = edges
        try:
            jx.write_json(path, doc)
        except OSError as e:
            QMessageBox.critical(self, "書き出し", f"書き出しに失敗しました：{e}")
            return

        html_path = None
        if self.export_multi_html_check.isChecked():
            html_path = os.path.splitext(path)[0] + ".html"
            try:
                html_export.write_standalone_html(html_path, doc)
            except OSError as e:
                QMessageBox.warning(self, "書き出し", f"JSONは書き出せましたが、HTMLの書き出しに失敗しました：{e}")
                html_path = None

        candidate_note = f"（{candidate.rank}位の案、2棟を含む）"
        if html_path:
            self.export_multi_msg.setText(f"書き出しました：{path}\n{html_path}{candidate_note}")
        else:
            self.export_multi_msg.setText(f"書き出しました：{path}{candidate_note}")
