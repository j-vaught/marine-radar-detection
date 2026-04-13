"""Interactive PPI crop / circular-mask tuner.

Usage:
    python tools/crop_tuner.py
    # then open http://localhost:8081 in a browser

Sliders adjust:
    cx / cy         — circle centre in pixels
    radius          — circular-mask radius
    crop left/right/top/bottom — outer crop-box offsets (in pixels from the
                                  respective image edges)

A "Save" button writes the current settings to
``tools/crop_config.json`` which the figure generator reads.
"""

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data/raw"
CFG_PATH = ROOT / "tools/crop_config.json"

DEFAULTS = {
    "dataset": "DAAN",
    "image": "342.png",
    "cx": 514,
    "cy": 508,
    "radius": 488,
    "left": 20,     # pixels from left edge to start crop
    "right": 20,    # pixels from right edge to end crop
    "top": 10,      # pixels from top edge
    "bottom": 10,   # pixels from bottom edge
    "draw_mask_ring": True,
}

app = Flask(__name__, static_folder=None)


# ── State ────────────────────────────────────────────────────────────
def load_cfg() -> dict:
    if CFG_PATH.exists():
        try:
            with open(CFG_PATH) as f:
                cfg = json.load(f)
            for k, v in DEFAULTS.items():
                cfg.setdefault(k, v)
            return cfg
        except Exception:
            pass
    return dict(DEFAULTS)


def save_cfg(cfg: dict) -> None:
    CFG_PATH.write_text(json.dumps(cfg, indent=2))


# ── Image processing ─────────────────────────────────────────────────
def _image_path(cfg: dict) -> Path:
    return RAW_DIR / cfg["dataset"] / "radarImages" / cfg["image"]


def _encode(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def render_preview(cfg: dict) -> dict:
    path = _image_path(cfg)
    if not path.exists():
        return {"error": f"not found: {path}"}

    raw = cv2.imread(str(path))   # BGR
    h, w = raw.shape[:2]

    cx = int(cfg["cx"])
    cy = int(cfg["cy"])
    r  = int(cfg["radius"])

    left   = max(0, int(cfg["left"]))
    right  = max(0, int(cfg["right"]))
    top    = max(0, int(cfg["top"]))
    bottom = max(0, int(cfg["bottom"]))

    x1 = left
    x2 = max(x1 + 1, w - right)
    y1 = top
    y2 = max(y1 + 1, h - bottom)

    # Overlay on raw to show where the circle + crop box sit
    overlay = raw.copy()
    # Crop box
    cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)
    # Circle
    cv2.circle(overlay, (cx, cy), r, (0, 10, 255), 2)
    # Centre crosshair
    cv2.drawMarker(overlay, (cx, cy), (0, 10, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

    # Produce the actual red-channel + masked result crop
    red = raw[:, :, 2].copy()
    mask = np.zeros_like(red)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    masked = np.where(mask > 0, red, np.zeros_like(red))

    # 3-channel tint so we can draw circle outline on crop
    masked_bgr = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    if cfg.get("draw_mask_ring", True):
        cv2.circle(masked_bgr, (cx, cy), r, (80, 80, 80), 1)

    overlay_crop = overlay[y1:y2, x1:x2]
    masked_crop  = masked_bgr[y1:y2, x1:x2]

    return {
        "width": w,
        "height": h,
        "overlay_full": _encode(overlay),
        "overlay_crop": _encode(overlay_crop),
        "masked_crop":  _encode(masked_crop),
        "crop_bounds":  [x1, y1, x2, y2],
    }


# ── Routes ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/config", methods=["GET"])
def api_get_config():
    return jsonify(load_cfg())


@app.route("/api/config", methods=["POST"])
def api_set_config():
    cfg = load_cfg()
    cfg.update(request.get_json(force=True))
    save_cfg(cfg)
    return jsonify(cfg)


@app.route("/api/datasets")
def api_datasets():
    out = {}
    for ds in ["DAAN", "DARC", "MANV"]:
        d = RAW_DIR / ds / "radarImages"
        if d.exists():
            out[ds] = sorted(p.name for p in d.glob("*.png"))[:120]
    return jsonify(out)


@app.route("/api/preview")
def api_preview():
    cfg = load_cfg()
    # Allow query params to override without saving
    for k in ("cx", "cy", "radius", "left", "right", "top", "bottom"):
        v = request.args.get(k)
        if v is not None:
            cfg[k] = int(v)
    for k in ("dataset", "image"):
        v = request.args.get(k)
        if v is not None:
            cfg[k] = v
    return jsonify(render_preview(cfg))


# ── Frontend ─────────────────────────────────────────────────────────
INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>Radar PPI Crop Tuner</title>
<style>
  :root {
    --garnet: #73000A;
    --rose:   #CC2E40;
    --atl:    #466A9F;
    --black:  #000000;
    --b90:    #363636;
    --b70:    #5C5C5C;
    --b50:    #A2A2A2;
    --b10:    #ECECEC;
    --sand:   #FFF2E3;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: var(--b10); color: var(--black);
               font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue",
               Arial, sans-serif; font-size: 14px; }
  header { background: var(--black); color: white; padding: 10px 18px;
           display: flex; justify-content: space-between; align-items: center; }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: .3px; }
  header .badge { background: var(--garnet); padding: 3px 8px; font-size: 11px;
                  text-transform: uppercase; letter-spacing: .6px; }

  main { display: grid;
         grid-template-columns: 320px 1fr;
         gap: 18px; padding: 18px; }

  .panel { background: white; border: 1px solid var(--b50);
           padding: 14px 16px; }
  .panel h2 { margin: 0 0 8px 0; font-size: 13px;
              text-transform: uppercase; letter-spacing: .7px;
              color: var(--garnet); border-bottom: 1px solid var(--garnet);
              padding-bottom: 4px; }

  label { display: flex; justify-content: space-between; font-weight: 600;
          margin: 10px 0 4px; font-size: 12px; color: var(--b90); }
  label span.val { font-variant-numeric: tabular-nums; color: var(--garnet); }
  input[type=range] { width: 100%; accent-color: var(--garnet); }
  select, input[type=text] { width: 100%; padding: 6px; font-size: 13px;
                              border: 1px solid var(--b50); background: white; }
  button { padding: 8px 14px; background: var(--garnet); color: white;
           border: none; cursor: pointer; font-weight: 600;
           letter-spacing: .4px; text-transform: uppercase; font-size: 12px; }
  button:hover { background: var(--rose); }
  button.ghost { background: white; color: var(--black);
                 border: 1px solid var(--b50); }
  button.ghost:hover { background: var(--b10); color: var(--garnet); }

  .buttons { display: flex; gap: 8px; margin-top: 12px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }

  .previews { display: grid; gap: 14px; }
  .img-card { background: white; border: 1px solid var(--b50); padding: 10px; }
  .img-card h3 { margin: 0 0 6px 0; font-size: 12px;
                 text-transform: uppercase; letter-spacing: .6px;
                 color: var(--b70); display: flex;
                 justify-content: space-between; align-items: baseline; }
  .img-card h3 .tag { color: var(--atl); font-size: 10px; }
  .img-card img { width: 100%; display: block; background: #000; }

  .small { font-size: 11px; color: var(--b70); margin-top: 4px; }
  .status { margin-top: 10px; padding: 6px 10px; background: var(--sand);
            border-left: 3px solid var(--garnet); font-size: 12px;
            color: var(--b90); }
  .swatch-circle { display: inline-block; width: 12px; height: 12px;
                   border-radius: 50%; background: #ff0a0a;
                   vertical-align: middle; margin-right: 4px; }
  .swatch-box { display: inline-block; width: 14px; height: 10px;
                background: #ffff00; border: 1px solid #000;
                vertical-align: middle; margin-right: 4px; }

  footer { text-align: center; padding: 10px; color: var(--b70);
           font-size: 11px; }
</style>
</head>
<body>

<header>
  <h1>Radar PPI Crop Tuner</h1>
  <span class="badge">live preview</span>
</header>

<main>
  <!-- Controls -->
  <section class="panel">
    <h2>Image</h2>
    <label>Dataset <span class="val" id="ds-val"></span></label>
    <select id="dataset"></select>

    <label>Frame <span class="val" id="img-val"></span></label>
    <select id="image"></select>

    <h2 style="margin-top:18px;">Circle</h2>
    <label>Centre X <span class="val" id="cx-val"></span></label>
    <input type="range" id="cx" min="400" max="650" step="1"/>
    <label>Centre Y <span class="val" id="cy-val"></span></label>
    <input type="range" id="cy" min="400" max="650" step="1"/>
    <label>Radius <span class="val" id="radius-val"></span></label>
    <input type="range" id="radius" min="350" max="560" step="1"/>

    <h2 style="margin-top:18px;">Crop (edge offsets)</h2>
    <div class="row">
      <div>
        <label>Left <span class="val" id="left-val"></span></label>
        <input type="range" id="left" min="0" max="400" step="1"/>
      </div>
      <div>
        <label>Right <span class="val" id="right-val"></span></label>
        <input type="range" id="right" min="0" max="400" step="1"/>
      </div>
      <div>
        <label>Top <span class="val" id="top-val"></span></label>
        <input type="range" id="top" min="0" max="400" step="1"/>
      </div>
      <div>
        <label>Bottom <span class="val" id="bottom-val"></span></label>
        <input type="range" id="bottom" min="0" max="400" step="1"/>
      </div>
    </div>

    <div class="buttons">
      <button id="save">Save to config</button>
      <button class="ghost" id="fit">Auto-fit circle</button>
      <button class="ghost" id="reset">Reset</button>
    </div>
    <div class="status" id="status">Move a slider to preview.</div>
    <div class="small">
      <span class="swatch-circle"></span> red circle = mask boundary.
      <span class="swatch-box"></span> yellow box = crop window.
    </div>
  </section>

  <!-- Previews -->
  <section class="previews">
    <div class="img-card">
      <h3>Full frame with overlay
          <span class="tag" id="size-tag">—</span></h3>
      <img id="overlay-full"/>
    </div>
    <div class="row" style="grid-template-columns:1fr 1fr;">
      <div class="img-card">
        <h3>Overlay crop <span class="tag">as shown in report</span></h3>
        <img id="overlay-crop"/>
      </div>
      <div class="img-card">
        <h3>Red-channel masked crop <span class="tag">pipeline output</span></h3>
        <img id="masked-crop"/>
      </div>
    </div>
  </section>
</main>

<footer>
  crop_config.json is written next to this script. The figure generator reads
  it on next run — re-run <code>python report/generate_figures.py</code> to
  refresh the PDF.
</footer>

<script>
const SLIDERS = ["cx","cy","radius","left","right","top","bottom"];
let state = {};

async function loadDatasets() {
  const r = await fetch("/api/datasets"); return r.json();
}
async function loadConfig() {
  const r = await fetch("/api/config"); return r.json();
}
async function saveConfig(cfg) {
  const r = await fetch("/api/config", {
    method:"POST", headers:{"content-type":"application/json"},
    body: JSON.stringify(cfg)
  });
  return r.json();
}

function qs(params) {
  return Object.entries(params).map(([k,v])=>`${k}=${encodeURIComponent(v)}`).join("&");
}

async function refresh() {
  document.getElementById("status").textContent = "Rendering…";
  const r = await fetch("/api/preview?" + qs(state));
  const data = await r.json();
  if (data.error) {
    document.getElementById("status").textContent = data.error;
    return;
  }
  document.getElementById("overlay-full").src = data.overlay_full;
  document.getElementById("overlay-crop").src = data.overlay_crop;
  document.getElementById("masked-crop").src  = data.masked_crop;
  document.getElementById("size-tag").textContent =
    `${data.width}×${data.height} | crop: ${data.crop_bounds.join(", ")}`;
  document.getElementById("status").textContent =
    `OK — crop box [${data.crop_bounds[0]}, ${data.crop_bounds[1]}] to ` +
    `[${data.crop_bounds[2]}, ${data.crop_bounds[3]}].`;
}

function bindSlider(id) {
  const el = document.getElementById(id);
  const lab = document.getElementById(id + "-val");
  el.addEventListener("input", () => {
    state[id] = +el.value;
    lab.textContent = el.value;
    refresh();
  });
}

function populateSelect(sel, items, current) {
  sel.innerHTML = "";
  items.forEach(v => {
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    if (v === current) o.selected = true;
    sel.appendChild(o);
  });
}

(async () => {
  const cfg = await loadConfig();
  const sets = await loadDatasets();
  state = { ...cfg };

  SLIDERS.forEach(id => {
    document.getElementById(id).value = cfg[id];
    document.getElementById(id + "-val").textContent = cfg[id];
    bindSlider(id);
  });

  const dsSel  = document.getElementById("dataset");
  const imgSel = document.getElementById("image");
  populateSelect(dsSel, Object.keys(sets), cfg.dataset);
  populateSelect(imgSel, sets[cfg.dataset] || [], cfg.image);
  document.getElementById("ds-val").textContent = cfg.dataset;
  document.getElementById("img-val").textContent = cfg.image;

  dsSel.addEventListener("change", () => {
    state.dataset = dsSel.value;
    document.getElementById("ds-val").textContent = dsSel.value;
    populateSelect(imgSel, sets[dsSel.value] || [], null);
    state.image = imgSel.value;
    document.getElementById("img-val").textContent = imgSel.value;
    refresh();
  });
  imgSel.addEventListener("change", () => {
    state.image = imgSel.value;
    document.getElementById("img-val").textContent = imgSel.value;
    refresh();
  });

  document.getElementById("save").addEventListener("click", async () => {
    const saved = await saveConfig(state);
    document.getElementById("status").textContent =
      `Saved → tools/crop_config.json (cx=${saved.cx}, cy=${saved.cy}, ` +
      `r=${saved.radius}, L=${saved.left}, R=${saved.right}, ` +
      `T=${saved.top}, B=${saved.bottom}).`;
  });

  document.getElementById("reset").addEventListener("click", async () => {
    const defaults = { dataset:"DAAN", image:"342.png",
                       cx:514, cy:508, radius:488,
                       left:20, right:20, top:10, bottom:10 };
    state = { ...defaults };
    SLIDERS.forEach(id => {
      document.getElementById(id).value = defaults[id];
      document.getElementById(id + "-val").textContent = defaults[id];
    });
    dsSel.value = defaults.dataset;
    populateSelect(imgSel, sets[defaults.dataset] || [], defaults.image);
    refresh();
  });

  // Auto-fit button: run a quick radial-profile measurement server-side.
  document.getElementById("fit").addEventListener("click", async () => {
    document.getElementById("status").textContent = "Auto-fitting…";
    const r = await fetch("/api/autofit?" + qs({dataset: state.dataset, image: state.image}));
    const d = await r.json();
    if (d.error) { document.getElementById("status").textContent = d.error; return; }
    state.cx = d.cx; state.cy = d.cy; state.radius = d.radius;
    ["cx","cy","radius"].forEach(id => {
      document.getElementById(id).value = state[id];
      document.getElementById(id + "-val").textContent = state[id];
    });
    refresh();
  });

  refresh();
})();
</script>
</body></html>
"""


# ── Auto-fit route ────────────────────────────────────────────────────
@app.route("/api/autofit")
def api_autofit():
    ds = request.args.get("dataset", "DAAN")
    img = request.args.get("image", "342.png")
    path = RAW_DIR / ds / "radarImages" / img
    if not path.exists():
        return jsonify({"error": f"not found: {path}"})

    raw = cv2.imread(str(path))
    red = raw[:, :, 2]
    h, w = red.shape

    # Centroid of bright red pixels (targets + outer ring)
    mask = red > 10
    if mask.sum() < 50:
        return jsonify({"error": "image too dark for auto-fit"})
    ys, xs = np.where(mask)
    cx = int(np.median(xs))
    cy = int(np.median(ys))

    # Radial mean-intensity profile → pick peak in outer annulus
    yy, xxm = np.indices(red.shape)
    d = np.sqrt((xxm - cx) ** 2 + (yy - cy) ** 2)
    max_r = int(min(cx, cy, w - cx, h - cy)) - 2
    mi = np.zeros(max_r)
    for r in range(max_r):
        ring = (d >= r - 0.5) & (d < r + 0.5)
        if ring.any():
            mi[r] = red[ring].mean()

    # Bright PPI outer ring is a peak in mi after r = 350
    search = mi.copy()
    search[:350] = 0
    peak = int(np.argmax(search))
    # Our mask should sit ≈ 4 px inside the peak ring to hide the ring itself.
    radius = max(380, peak - 4)

    return jsonify({"cx": cx, "cy": cy, "radius": radius, "peak": peak})


if __name__ == "__main__":
    print("Crop tuner running at http://localhost:8081")
    print(f"Config file: {CFG_PATH}")
    app.run(host="0.0.0.0", port=8081, debug=False)
