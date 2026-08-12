import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from volume_finder.io import html_export
from volume_finder.io import json_export as jx


def _sample_doc():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    edges = [{"type": "road", "width": 8.0}] + [{"type": "rinchi", "width": 0}] * 3
    return jx.build_json(square, 34.6937, 135.5023, edges)


def test_template_file_exists():
    assert os.path.exists(html_export.TEMPLATE_PATH)


def test_build_standalone_html_embeds_valid_json_and_apply_call():
    doc = _sample_doc()
    html = html_export.build_standalone_html(doc)
    assert '<script type="application/json" id="__volume_finder_data__">' in html
    assert "apply(JSON.parse(raw))" in html
    assert html.rstrip().endswith("</html>")

    start = html.index('id="__volume_finder_data__">') + len('id="__volume_finder_data__">')
    end = html.index("</script>", start)
    embedded_text = html[start:end].replace("<\\/", "</")
    embedded = json.loads(embedded_text)
    assert embedded["format"] == "hikage-checker"
    assert embedded["site"] == doc["site"]


def test_build_standalone_html_escapes_closing_script_tags_in_data():
    doc = _sample_doc()
    doc["_qgis"]["warnings"] = ["</script><script>alert(1)</script>"]
    html = html_export.build_standalone_html(doc)
    # the raw (unescaped) sequence must never appear inside the injected block,
    # or it would prematurely close the <script> tag in a real browser
    start = html.index('id="__volume_finder_data__">')
    end = html.index("</script>", start)
    injected_block = html[start:end]
    assert "</script>" not in injected_block
    assert "<\\/script>" in injected_block


def test_build_standalone_html_injected_before_closing_body():
    doc = _sample_doc()
    html = html_export.build_standalone_html(doc)
    body_close = html.rindex("</body>")
    data_marker = html.index('id="__volume_finder_data__">')
    assert data_marker < body_close


def test_write_standalone_html_roundtrip(tmp_path):
    doc = _sample_doc()
    out_path = str(tmp_path / "out.html")
    html_export.write_standalone_html(out_path, doc)
    with open(out_path, encoding="utf-8") as f:
        content = f.read()
    assert "__volume_finder_data__" in content
    assert content.rstrip().endswith("</html>")
