"""Interactive AIS-to-radar alignment tuner.

Usage:
    python tools/ais_tuner.py
    # then open http://localhost:8082 in a browser

Lets you manually dial in the polar projection parameters per dataset
by watching AIS seeds (orange dots) snap onto bright radar blobs
(outlined atlantic). Good for spotting:

    * centre offset (cx, cy)
    * wrong range scale (max_range_m)
    * global azimuth rotation (angle_deg)
    * east/west mirroring of θ (flip_theta)

On "Save to config" the current values for the active dataset are
written to ``tools/ais_calibration.json``.  The dataset generator can
optionally read this to override ``DATASET_CALIBRATION`` defaults.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running this script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, Response, jsonify, request   # noqa: E402

from src.data.ais_anchored_gt import (                # noqa: E402
    parse_frame_annotations,
    split_ais_to_seeds,
    extract_blobs,
)
from src.data.radar_mask import (                     # noqa: E402
    DATASET_CALIBRATION,
    POLAR_RADIUS_PX,
    create_radar_mask,
)


RAW_DIR = ROOT / "data/raw"
CFG_PATH = ROOT / "tools/ais_calibration.json"

DATASETS = ["DAAN", "DARC", "MANV"]

DEFAULTS = {
    "dataset": "MANV",
    "image": "100.png",
    "cx": 514,
    "cy": 528,
    "radius": 466,        # visual PPI radius in pixels
    "max_range_m": 2800,  # metres at full radius
    "angle_deg": 0.0,     # extra azimuth rotation applied after polar->cart
    "flip_theta": False,  # if the radar wraps clockwise-left instead of CW-right
    "match_radius_px": 20,
}


app = Flask(__name__, static_folder=None)


# ── State ────────────────────────────────────────────────────────────
def _initial_cfg_for(ds: str) -> dict:
    src = DATASET_CALIBRATION[ds]
    cfg = dict(DEFAULTS)
    cfg["dataset"] = ds
    cfg["cx"] = src["cx"]
    cfg["cy"] = src["cy"]
    cfg["radius"] = src["r_px_max"]
    cfg["max_range_m"] = src["max_range_m"]
    return cfg


def load_cfg() -> dict:
    if CFG_PATH.exists():
        try:
            return json.loads(CFG_PATH.read_text())
        except Exception:
            pass
    return {ds: _initial_cfg_for(ds) for ds in DATASETS}


def save_cfg(full: dict) -> None:
    CFG_PATH.write_text(json.dumps(full, indent=2))


# ── Image processing ─────────────────────────────────────────────────
def _encode(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def project_seeds(polar_entry, params) -> list[tuple[int, int, int, str]]:
    """Apply current params to polar points → list of (px, py, obj_id, cluster)."""
    objs = parse_frame_annotations(polar_entry or [])
    cx, cy, rmax_px, rmax_m = (params["cx"], params["cy"],
                                params["radius"], params["max_range_m"])
    angle = np.deg2rad(params["angle_deg"])
    flip = params["flip_theta"]

    seeds = split_ais_to_seeds(objs, cx=cx, cy=cy,
                                r_px_max=rmax_px, max_range_m=rmax_m)
    rotated: list[tuple[int, int, int, int]] = []
    for s in seeds:
        sx, sy = s.centroid_px
        # Apply extra angle rotation / θ flip around (cx, cy)
        dx = sx - cx
        dy = sy - cy
        if flip:
            dx = -dx
        if angle != 0.0:
            c, si = np.cos(angle), np.sin(angle)
            dx, dy = c * dx - si * dy, si * dx + c * dy
        rotated.append((int(cx + dx), int(cy + dy), s.obj_id, s.cluster_idx))
    return rotated


def render_preview(params) -> dict:
    ds = params["dataset"]
    img_path = RAW_DIR / ds / "radarImages" / params["image"]
    if not img_path.exists():
        return {"error": f"not found: {img_path}"}

    bgr = cv2.imread(str(img_path))
    h, w = bgr.shape[:2]

    polar_path = RAW_DIR / ds / "polarMeas.json"
    polar = json.loads(polar_path.read_text())
    entry = polar.get(str(int(params["image"].split(".")[0])))

    mask = create_radar_mask(h, w)
    blobs = extract_blobs(bgr, radar_mask=mask, dataset=ds)
    seeds = project_seeds(entry, params)

    # Metrics
    blob_xy = np.array([[b.cx, b.cy] for b in blobs]) if blobs else np.zeros((0, 2))
    match_r = float(params["match_radius_px"])
    matched = unmatched = 0
    total_dist: list[float] = []
    if len(blob_xy):
        for (sx, sy, _oid, _ci) in seeds:
            d = np.hypot(blob_xy[:, 0] - sx, blob_xy[:, 1] - sy)
            nearest = float(d.min())
            total_dist.append(nearest)
            if nearest <= match_r:
                matched += 1
            else:
                unmatched += 1

    # Overlay on the raw image (keep original colour)
    overlay = bgr.copy()
    # 1. draw the radar mask outline (thin grey ring)
    cv2.circle(overlay, (params["cx"], params["cy"]),
               params["radius"], (80, 80, 80), 1)
    # 2. draw all blobs in atlantic
    for b in blobs:
        cv2.rectangle(overlay, (b.x1, b.y1), (b.x2, b.y2),
                      (159, 106, 70), 1)
    # 3. draw AIS seeds: orange if matched, red if not
    for (sx, sy, oid, ci) in seeds:
        if not len(blob_xy):
            colour = (0, 0, 255)
        else:
            d = np.hypot(blob_xy[:, 0] - sx, blob_xy[:, 1] - sy).min()
            colour = (0, 165, 255) if d <= match_r else (0, 0, 255)
        cv2.circle(overlay, (sx, sy), 5, colour, -1)
        cv2.circle(overlay, (sx, sy), 5, (0, 0, 0), 1)
        # Match radius ring for the first seed only to keep it clean
    # Draw a single match-radius preview ring at the first seed
    if seeds:
        sx, sy, _, _ = seeds[0]
        cv2.circle(overlay, (sx, sy), int(match_r), (120, 200, 255), 1)
    # 4. cross-hair at polar origin
    cx0, cy0 = params["cx"], params["cy"]
    cv2.drawMarker(overlay, (cx0, cy0), (255, 255, 255),
                   cv2.MARKER_CROSS, 20, 2)

    return {
        "overlay": _encode(overlay),
        "n_seeds": len(seeds),
        "n_blobs": len(blobs),
        "n_matched": int(matched),
        "n_unmatched": int(unmatched),
        "dist_median": float(np.median(total_dist)) if total_dist else 0.0,
        "dist_p75":   float(np.percentile(total_dist, 75)) if total_dist else 0.0,
        "dist_max":   float(max(total_dist)) if total_dist else 0.0,
    }


# ── Routes ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/config", methods=["GET"])
def api_get_config():
    cfg = load_cfg()
    # Ensure every dataset has full field set
    for ds in DATASETS:
        cfg.setdefault(ds, {})
        for k, v in DEFAULTS.items():
            cfg[ds].setdefault(k, v)
        cfg[ds]["dataset"] = ds
        # pull in DATASET_CALIBRATION-derived defaults if missing
        src = DATASET_CALIBRATION[ds]
        cfg[ds].setdefault("cx", src["cx"])
        cfg[ds].setdefault("cy", src["cy"])
        cfg[ds].setdefault("radius", src["r_px_max"])
        cfg[ds].setdefault("max_range_m", src["max_range_m"])
    return jsonify(cfg)


@app.route("/api/config", methods=["POST"])
def api_set_config():
    body = request.get_json(force=True)
    cfg = load_cfg()
    ds = body.get("dataset", "DAAN")
    cfg[ds] = dict(DEFAULTS, **body)
    save_cfg(cfg)
    return jsonify(cfg)


@app.route("/api/datasets")
def api_datasets():
    out = {}
    for ds in DATASETS:
        d = RAW_DIR / ds / "radarImages"
        if d.exists():
            out[ds] = sorted(p.name for p in d.glob("*.png"))[:200]
    return jsonify(out)


@app.route("/api/preview")
def api_preview():
    params = dict(DEFAULTS)
    for k in ("cx", "cy", "radius", "match_radius_px"):
        v = request.args.get(k)
        if v is not None:
            params[k] = int(float(v))
    for k in ("max_range_m", "angle_deg"):
        v = request.args.get(k)
        if v is not None:
            params[k] = float(v)
    for k in ("dataset", "image"):
        v = request.args.get(k)
        if v is not None:
            params[k] = v
    v = request.args.get("flip_theta")
    if v is not None:
        params["flip_theta"] = v.lower() in ("1", "true", "yes", "on")
    return jsonify(render_preview(params))


# ── Frontend ────────────────────────────────────────────────────────
INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>AIS Alignment Tuner</title>
<style>
  :root {
    --garnet: #73000A;
    --rose:   #CC2E40;
    --atl:    #466A9F;
    --black:  #000;
    --b90:    #363636;
    --b70:    #5C5C5C;
    --b50:    #A2A2A2;
    --b10:    #ECECEC;
    --sand:   #FFF2E3;
  }
  *  { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; background: var(--b10); color: var(--black);
    font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    font-size: 13px;
  }
  header {
    background: var(--black); color: white; padding: 10px 18px;
    display: flex; align-items: center; justify-content: space-between;
  }
  header h1 { margin: 0; font-size: 15px; letter-spacing: .4px;
              font-weight: 600; }
  header .badge { background: var(--garnet); padding: 3px 9px;
                  font-size: 10px; text-transform: uppercase;
                  letter-spacing: .6px; }
  main { display: grid; grid-template-columns: 320px 1fr;
         gap: 16px; padding: 16px; }
  .panel { background: white; border: 1px solid var(--b50);
           padding: 12px 14px; }
  .panel h2 { margin: 0 0 6px 0; font-size: 11px;
              letter-spacing: .8px; text-transform: uppercase;
              color: var(--garnet); border-bottom: 1px solid var(--garnet);
              padding-bottom: 3px; }
  label { display: flex; justify-content: space-between;
          font-weight: 600; margin: 10px 0 3px;
          font-size: 11px; color: var(--b90); }
  label span.val { color: var(--garnet);
                   font-variant-numeric: tabular-nums; }
  input[type=range] { width: 100%; accent-color: var(--garnet); }
  select { width: 100%; padding: 5px; font-size: 12px;
           border: 1px solid var(--b50); background: white; }
  .checkbox-row { display: flex; justify-content: space-between;
                  align-items: center; margin: 8px 0; font-size: 12px;
                  font-weight: 600; }
  button { padding: 7px 12px; background: var(--garnet); color: white;
           border: none; cursor: pointer; font-weight: 600;
           letter-spacing: .3px; text-transform: uppercase;
           font-size: 11px; }
  button:hover { background: var(--rose); }
  button.ghost { background: white; color: black;
                 border: 1px solid var(--b50); }
  button.ghost:hover { background: var(--b10); color: var(--garnet); }
  .buttons { display: flex; gap: 6px; margin-top: 12px; flex-wrap: wrap; }
  .status { margin-top: 10px; padding: 7px 10px;
            background: var(--sand); border-left: 3px solid var(--garnet);
            font-size: 11px; color: var(--b90);
            font-variant-numeric: tabular-nums; }
  .metrics {
    display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px;
    margin-top: 8px; font-variant-numeric: tabular-nums;
  }
  .metrics .m-key { color: var(--b70); font-size: 10px;
                    text-transform: uppercase; letter-spacing: .4px; }
  .metrics .m-val { font-size: 14px; font-weight: 600; }
  .m-good { color: var(--atl); }
  .m-bad  { color: var(--garnet); }
  .preview { background: white; border: 1px solid var(--b50);
             padding: 10px; }
  .preview h3 { margin: 0 0 6px 0; font-size: 11px;
                letter-spacing: .6px; text-transform: uppercase;
                color: var(--b70); display: flex; justify-content: space-between;
                align-items: baseline; }
  .preview img { width: 100%; display: block; background: #000; }
  .legend { font-size: 10px; color: var(--b70); padding: 6px 0; }
  .dot { display: inline-block; width: 10px; height: 10px;
         border-radius: 50%; vertical-align: middle; margin-right: 3px;
         border: 1px solid #000; }
  .dot.o { background: #ffa500; }  /* orange = matched */
  .dot.r { background: #ff0a0a; }  /* red    = unmatched */
  footer { text-align: center; padding: 8px; color: var(--b70);
           font-size: 10px; }
</style>
</head>
<body>

<header>
  <h1>AIS-to-Radar Alignment Tuner</h1>
  <span class="badge">live preview · port 8082</span>
</header>

<main>
  <section class="panel">
    <h2>Dataset</h2>
    <label>Dataset</label>
    <select id="dataset"></select>
    <label>Frame</label>
    <select id="image"></select>

    <h2 style="margin-top:14px;">Polar origin</h2>
    <label>cx <span class="val" id="cx-val"></span></label>
    <input type="range" id="cx" min="400" max="650" step="1">
    <label>cy <span class="val" id="cy-val"></span></label>
    <input type="range" id="cy" min="400" max="650" step="1">

    <h2 style="margin-top:14px;">Scale</h2>
    <label>PPI radius (px) <span class="val" id="radius-val"></span></label>
    <input type="range" id="radius" min="300" max="560" step="1">
    <label>Max range (m) <span class="val" id="max_range_m-val"></span></label>
    <input type="range" id="max_range_m" min="1500" max="8000" step="25">

    <h2 style="margin-top:14px;">Rotation</h2>
    <label>Angle offset (°) <span class="val" id="angle_deg-val"></span></label>
    <input type="range" id="angle_deg" min="-30" max="30" step="0.25">
    <div class="checkbox-row">
      <label style="margin: 0;">Flip θ (E/W mirror)</label>
      <input type="checkbox" id="flip_theta">
    </div>

    <h2 style="margin-top:14px;">Match radius (viz)</h2>
    <label>Threshold (px) <span class="val" id="match_radius_px-val"></span></label>
    <input type="range" id="match_radius_px" min="5" max="120" step="1">

    <div class="buttons">
      <button id="save">Save dataset</button>
      <button class="ghost" id="reset">Reset</button>
    </div>

    <div class="metrics">
      <div class="m-key">seeds</div> <div class="m-val" id="m-seeds">—</div>
      <div class="m-key">blobs</div> <div class="m-val" id="m-blobs">—</div>
      <div class="m-key">matched</div>   <div class="m-val m-good" id="m-match">—</div>
      <div class="m-key">unmatched</div> <div class="m-val m-bad"  id="m-unm">—</div>
      <div class="m-key">dist median</div> <div class="m-val" id="m-med">—</div>
      <div class="m-key">dist p75</div>    <div class="m-val" id="m-p75">—</div>
    </div>

    <div class="status" id="status">Move a slider to preview.</div>
    <div class="legend">
      <span class="dot o"></span> matched AIS seed &nbsp;
      <span class="dot r"></span> unmatched AIS seed &nbsp;
      blue rectangles = image blobs
    </div>
  </section>

  <section class="preview">
    <h3>Radar image + AIS overlay
      <span id="size-tag">—</span></h3>
    <img id="overlay">
  </section>
</main>
<footer>
  tools/ais_calibration.json keeps per-dataset values.
</footer>

<script>
const KEYS_INT   = ["cx","cy","radius","match_radius_px"];
const KEYS_FLOAT = ["max_range_m","angle_deg"];
const ALL_SLIDERS = [...KEYS_INT, ...KEYS_FLOAT];
let CFG = {};
let DS  = "MANV";
let IMG = "100.png";
let DATASETS = {};

async function loadConfig() {
  const r = await fetch("/api/config");
  return r.json();
}
async function saveConfig(values) {
  await fetch("/api/config", {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify(values)
  });
}

function qs(params) {
  return Object.entries(params)
    .map(([k,v]) => `${k}=${encodeURIComponent(v)}`).join("&");
}

function current() {
  return { ...CFG[DS], dataset: DS, image: IMG };
}

function applySlidersFromState() {
  const s = current();
  for (const k of ALL_SLIDERS) {
    const el = document.getElementById(k);
    el.value = s[k];
    document.getElementById(k + "-val").textContent =
      typeof s[k] === "number" ? s[k] : s[k];
  }
  document.getElementById("flip_theta").checked = !!s.flip_theta;
}

async function refresh() {
  document.getElementById("status").textContent = "Rendering…";
  const params = current();
  const r = await fetch("/api/preview?" + qs(params));
  const d = await r.json();
  if (d.error) {
    document.getElementById("status").textContent = d.error;
    return;
  }
  document.getElementById("overlay").src = d.overlay;
  document.getElementById("size-tag").textContent =
    `${DS}/${IMG}  ·  cx=${params.cx}  cy=${params.cy}  ` +
    `range=${params.max_range_m}m  angle=${params.angle_deg.toFixed(2)}°` +
    (params.flip_theta ? "  FLIPPED" : "");
  document.getElementById("m-seeds").textContent = d.n_seeds;
  document.getElementById("m-blobs").textContent = d.n_blobs;
  document.getElementById("m-match").textContent = d.n_matched;
  document.getElementById("m-unm").textContent   = d.n_unmatched;
  document.getElementById("m-med").textContent   = d.dist_median.toFixed(1);
  document.getElementById("m-p75").textContent   = d.dist_p75.toFixed(1);
  document.getElementById("status").textContent =
    `OK — ${d.n_matched}/${d.n_seeds} seeds matched at ≤ ` +
    `${params.match_radius_px} px.`;
}

function bindSlider(k, isFloat) {
  const el = document.getElementById(k);
  const lab = document.getElementById(k + "-val");
  el.addEventListener("input", () => {
    CFG[DS][k] = isFloat ? parseFloat(el.value) : parseInt(el.value);
    lab.textContent = CFG[DS][k];
    refresh();
  });
}

function populateSelect(sel, items, current) {
  sel.innerHTML = "";
  for (const v of items) {
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    if (v === current) o.selected = true;
    sel.appendChild(o);
  }
}

(async () => {
  CFG = await loadConfig();
  const sets = await (await fetch("/api/datasets")).json();
  DATASETS = sets;

  DS = "MANV";
  populateSelect(document.getElementById("dataset"), Object.keys(sets), DS);
  populateSelect(document.getElementById("image"), sets[DS], "100.png");
  IMG = "100.png";

  KEYS_INT.forEach(k => bindSlider(k, false));
  KEYS_FLOAT.forEach(k => bindSlider(k, true));
  document.getElementById("flip_theta").addEventListener("change", e => {
    CFG[DS].flip_theta = e.target.checked;
    refresh();
  });

  document.getElementById("dataset").addEventListener("change", e => {
    DS = e.target.value;
    populateSelect(document.getElementById("image"), sets[DS], sets[DS][0]);
    IMG = sets[DS][0];
    applySlidersFromState();
    refresh();
  });
  document.getElementById("image").addEventListener("change", e => {
    IMG = e.target.value;
    refresh();
  });

  document.getElementById("save").addEventListener("click", async () => {
    await saveConfig(current());
    document.getElementById("status").textContent =
      `Saved ${DS} calibration -> tools/ais_calibration.json`;
  });

  document.getElementById("reset").addEventListener("click", async () => {
    CFG = await loadConfig();
    applySlidersFromState();
    refresh();
  });

  applySlidersFromState();
  refresh();
})();
</script>
</body></html>
"""


if __name__ == "__main__":
    print("AIS tuner running at http://localhost:8082")
    print("Per-dataset calibration is written to:", CFG_PATH)
    app.run(host="0.0.0.0", port=8082, debug=False)
