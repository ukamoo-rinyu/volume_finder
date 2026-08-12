"""プラグイン本体。メニュー登録とドックの表示切替のみを担当する。"""

import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .ui.dock import VolumeFinderDock

PLUGIN_NAME = "建築規制ボリューム検討"


class VolumeFinderPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dock = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        self.action = QAction(icon, PLUGIN_NAME, self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.triggered.connect(self.toggle_dock)
        self.iface.addPluginToMenu(PLUGIN_NAME, self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        self.iface.removePluginMenu(PLUGIN_NAME, self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.dock is not None:
            self.iface.removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = None

    def toggle_dock(self, checked):
        if self.dock is None:
            self.dock = VolumeFinderDock(self.iface, self.iface.mainWindow())
            self.dock.visibilityChanged.connect(self._on_dock_visibility_changed)
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.setVisible(checked)

    def _on_dock_visibility_changed(self, visible):
        if self.action is not None:
            self.action.setChecked(visible)
