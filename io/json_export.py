"""Writer for the HTML tool's `hikage-checker` JSON format (設計書 5.2).

Only the keys the HTML tool's apply() reads are put at the top level;
everything the plugin adds lives under `_qgis` so the HTML tool can
ignore it unmodified (設計書 1.2).
"""

import datetime
import json
from typing import Iterable, List, Optional, Sequence, Tuple

FORMAT = "hikage-checker"
VERSION = 4

# Defaults mirror the HTML tool's own defaults (大阪市 / 北緯35度冬至日).
DEFAULT_ZONE = "1住"
DEFAULT_FAR = 200
DEFAULT_BCR = 60
DEFAULT_BCRX = "0"
DEFAULT_CALC = {
    "mh": 4,
    "dt": 2,
    "autolv": True,
    "lv3": "3",
    "lv5": "5",
    "base": "pale",
}

VALID_EDGE_TYPES = {"road", "sui", "koen", "rinchi"}

DEFAULT_VIEW_ZOOM = 18  # HTMLツールの地図初期化と同じズーム(setup: zoom:18)


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


def build_edges(edges: Iterable[dict]) -> List[dict]:
    out = []
    for e in edges:
        etype = e.get("type", "rinchi")
        if etype not in VALID_EDGE_TYPES:
            etype = "rinchi"
        out.append({"type": etype, "width": float(e.get("width") or 0)})
    return out


def build_json(
    site_local: Sequence[Tuple[float, float]],
    origin_lat: float,
    origin_lng: float,
    edges: Iterable[dict],
    zone: str = DEFAULT_ZONE,
    far: float = DEFAULT_FAR,
    bcr: float = DEFAULT_BCR,
    bcrx: str = DEFAULT_BCRX,
    lat0: float = 35,
    buildings: Optional[List[dict]] = None,
    qgis_extra: Optional[dict] = None,
) -> dict:
    """Build a dict ready for json.dumps(), matching format:"hikage-checker".

    `site_local` must already be CCW-normalized local XY (meters,
    origin at the site centroid) — see core.geometry.to_ccw /
    latlng_to_local. `edges[i]` corresponds to the edge from
    site_local[i] to site_local[i+1] (設計書 2.1.2).
    """
    edges_list = list(edges)
    if len(edges_list) != len(site_local):
        raise ValueError(
            f"edges count ({len(edges_list)}) must match site vertex count ({len(site_local)})"
        )
    doc = {
        "format": FORMAT,
        "version": VERSION,
        "saved": _now_iso(),
        "origin": {"lat": origin_lat, "lng": origin_lng},
        "lat0": lat0,
        "zone": zone,
        "far": far,
        "bcr": bcr,
        "bcrx": bcrx,
        "site": [[round(x, 3), round(y, 3)] for x, y in site_local],
        "edges": build_edges(edges_list),
        "buildings": buildings or [],
        "calc": dict(DEFAULT_CALC),
        # HTML側のapply()はd.viewがあれば地図をそこに合わせる。省くと
        # 大阪市役所付近（アプリの既定中心）のままになってしまうため、
        # 必ず敷地の位置を入れておく（設計書1.3のHTML側での確認をスムーズに）。
        "view": {"center": [origin_lat, origin_lng], "zoom": DEFAULT_VIEW_ZOOM},
    }
    qgis_block = {
        "note": "以下はプラグインが付加した情報。HTMLでは使用しない",
        "searched_at": _now_iso(),
    }
    if qgis_extra:
        qgis_block.update(qgis_extra)
    doc["_qgis"] = qgis_block
    return doc


def write_json(path: str, doc: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
