"""探索をQGIS本体を固まらせずに実行するための QgsTask ラッパー (設計書 4.2)。

core.search はQGISに依存しない純粋な関数なので、ここでは進捗・中断・
例外の橋渡しだけを行う。QgsTask.run() はワーカースレッドで動くが、
pyqtSignal はQtのキュー接続で自動的にメインスレッドへ届く。
"""

from qgis.core import QgsTask
from qgis.PyQt.QtCore import pyqtSignal

from ..core import search as srch


class SearchTask(QgsTask):
    progress_update = pyqtSignal(int, int, object)  # done, total, best_candidate_or_None

    def __init__(self, description, site_pts, edges, zone, far, bcr, params):
        super().__init__(description, QgsTask.CanCancel)
        self.site_pts = site_pts
        self.edges = edges
        self.zone = zone
        self.far = far
        self.bcr = bcr
        self.params = params
        self.results = []
        self.pool = []  # search()のpool_out: 正規化済みスコア付きの候補プール（重み変更時の再ランキング用）
        self.exception = None

    def run(self):
        try:
            def progress_cb(done, total, best):
                if total:
                    self.setProgress(done / total * 100)
                self.progress_update.emit(done, total, best)

            self.results = srch.search(
                self.site_pts, self.edges, self.zone, self.far, self.bcr,
                params=self.params, progress_callback=progress_cb, should_stop=self.isCanceled,
                pool_out=self.pool,
            )
            return True
        except Exception as e:  # noqa: BLE001 - 例外はUI側に伝えて表示する
            self.exception = e
            return False
