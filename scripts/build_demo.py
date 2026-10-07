"""Inline the JS simulator and results into one self-contained demo page.

    python scripts/build_demo.py                      # -> docs/index.html (GitHub Pages)
    python scripts/build_demo.py --fragment out.html  # body-only variant for hosts that add <head>
"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--fragment")
args = ap.parse_args()

res = json.loads((ROOT / "docs/results.json").read_text())
models = {}
for p in sorted((ROOT / "curves").glob("*.json")):
    m = json.loads(p.read_text().replace("Infinity", "1e9"))
    models[p.stem] = {"edges": m["edges"], "h": m["h"], "horizon": m["horizon"], "kind": m.get("kind", "ruin")}
data = {"models": models, **{k: res[k] for k in ("curves", "sweep", "sweep_N", "capacity", "phase", "placement")}}

body = (ROOT / "docs/src/template.html").read_text()
body = body.replace("/*SIM_JS*/", (ROOT / "docs/src/sim.js").read_text())
body = body.replace("/*DATA_JSON*/", json.dumps(data, separators=(",", ":")))
if args.fragment:
    pathlib.Path(args.fragment).write_text(body)
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="Interactive simulator: how many VLA robots can one GPU serve '
        'under deadline and staleness SLOs?">\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
(ROOT / "docs/index.html").write_text(page)
print(f"docs/index.html: {len(page) / 1024:.0f} KB")
