import os
import sys

# insert the repo root (parent of volume_finder/) so imports are
# package-qualified (volume_finder.io...) and never shadow stdlib `io`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from volume_finder.core import geometry as geo
from volume_finder.io import json_export as jx


def test_build_json_roundtrips_ccw_site_and_edges():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    edges = [
        {"type": "road", "width": 8.0},
        {"type": "rinchi", "width": 0},
        {"type": "sui", "width": 3.0},
        {"type": "koen", "width": 0},
    ]
    doc = jx.build_json(square, 34.6937, 135.5023, edges)
    assert doc["format"] == "hikage-checker"
    assert doc["version"] == 4
    assert doc["origin"] == {"lat": 34.6937, "lng": 135.5023}
    assert doc["site"] == [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert doc["edges"][0] == {"type": "road", "width": 8.0}
    assert doc["zone"] == "1住"
    assert doc["far"] == 200
    assert "_qgis" in doc
    assert doc["_qgis"]["note"]


def test_build_json_rejects_mismatched_edge_count():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    with pytest.raises(ValueError):
        jx.build_json(square, 34.6937, 135.5023, [{"type": "road", "width": 8}])


def test_build_edges_falls_back_to_rinchi_on_unknown_type():
    out = jx.build_edges([{"type": "bogus", "width": 5}])
    assert out == [{"type": "rinchi", "width": 5.0}]


def test_build_json_qgis_extra_merges():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    edges = [{"type": "rinchi", "width": 0}] * 4
    doc = jx.build_json(
        square, 34.6937, 135.5023, edges, qgis_extra={"flipped": True, "source_fids": [1, 2]}
    )
    assert doc["_qgis"]["flipped"] is True
    assert doc["_qgis"]["source_fids"] == [1, 2]


def test_site_matches_ccw_normalized_geometry():
    cw_square = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    ccw_pts, flipped = geo.to_ccw(cw_square)
    edges = [{"type": "rinchi", "width": 0}] * len(ccw_pts)
    doc = jx.build_json(ccw_pts, 34.6937, 135.5023, edges, qgis_extra={"flipped": flipped})
    assert geo.signed_area([tuple(p) for p in doc["site"]]) > 0
