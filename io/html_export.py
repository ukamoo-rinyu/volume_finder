"""HTMLツールに直接読み込ませた状態の単体HTMLを書き出す（任意オプション）。

設計書1.3の想定フロー「気になる案をJSON書き出し → HTMLツールで開いて
確認」を1手で済ませたい場合に使う。同梱の hikage-osaka-v4_23.html を
テンプレートとして読み込み、</body>の直前にJSONデータを埋め込んだ
<script>を追加するだけ。埋め込んだデータはHTML側の「開く」ボタンが
使う apply() にそのまま渡すので、挙動は手動でJSONを読み込んだ場合と
完全に同じになる。
"""

import json
import os

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "html_template", "hikage-osaka-v4_23.html")


def build_standalone_html(doc: dict, template_path: str = TEMPLATE_PATH) -> str:
    """doc（json_export.build_json()の戻り値）を埋め込んだHTML文字列を組み立てる。"""
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # <script>タグ内に "</script" が現れると途中でタグが閉じてしまうため、
    # JSON中に紛れ込んでいても安全なように "</" を全てエスケープしておく
    # （<script type="application/json"> であってもブラウザのHTML
    # パーサーは中身を見ずに "</script" を探すため、これが唯一安全な方法）。
    json_text = json.dumps(doc, ensure_ascii=False).replace("</", "<\\/")

    injection = (
        '\n<script type="application/json" id="__volume_finder_data__">'
        f"{json_text}"
        "</script>\n"
        "<script>\n"
        "(function(){\n"
        '  try {\n'
        '    var raw = document.getElementById("__volume_finder_data__").textContent;\n'
        "    apply(JSON.parse(raw));\n"
        '    document.getElementById("jsonmsg").innerHTML='
        '"<b style=\\"color:var(--ok)\\">QGISプラグインの出力を読み込みました。</b>";\n'
        "  } catch (e) {\n"
        '    console.error("volume_finder: 自動読み込みに失敗しました", e);\n'
        "  }\n"
        "})();\n"
        "</script>\n"
    )

    marker = "</body>"
    idx = template.rfind(marker)
    if idx == -1:
        raise ValueError("テンプレートHTMLに</body>が見つかりません。テンプレートが壊れていないか確認してください。")
    return template[:idx] + injection + template[idx:]


def write_standalone_html(path: str, doc: dict, template_path: str = TEMPLATE_PATH) -> None:
    html = build_standalone_html(doc, template_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
