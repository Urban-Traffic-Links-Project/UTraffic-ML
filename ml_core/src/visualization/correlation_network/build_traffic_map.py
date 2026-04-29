"""
build_traffic_map.py
====================
Đọc toàn bộ file JSON từ ml_core/data/raw/tomtom_stats_frc5/
rồi nhúng dữ liệu vào file HTML để mở trực tiếp bằng trình duyệt (file://).

Cách dùng:
    python build_traffic_map.py

Output:
    traffic_map_embedded.html  (mở file này bằng Chrome/Firefox)

Đặt script này ở thư mục gốc project (cùng cấp với ml_core/).
"""

import json
import os
import glob
from pathlib import Path

# ── CẤU HÌNH ────────────────────────────────────────────────────────────────
DATA_DIR = Path("ml_core/data/raw/tomtom_stats_frc7")
OUTPUT_HTML = Path("traffic_map_embedded.html")
# ────────────────────────────────────────────────────────────────────────────


def load_all_segments(data_dir: Path) -> list:
    """Đọc tất cả job_xxxx_results.json, trích xuất segmentResults."""
    json_files = sorted(glob.glob(str(data_dir / "job_*_results.json")))
    if not json_files:
        raise FileNotFoundError(f"Không tìm thấy file JSON trong: {data_dir}")

    all_segments = []
    seen_ids = set()

    for fpath in json_files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            network = data.get("network", {})
            seg_results = network.get("segmentResults", [])

            for seg in seg_results:
                sid = seg.get("segmentId") or seg.get("newSegmentId")
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)

                # Chỉ giữ các trường cần thiết để giảm dung lượng HTML
                clean_seg = {
                    "segmentId": seg.get("segmentId"),
                    "speedLimit": seg.get("speedLimit"),
                    "frc": seg.get("frc"),
                    "streetName": seg.get("streetName", "Unknown"),
                    "distance": seg.get("distance", 0),
                    "shape": seg.get("shape", []),
                    "segmentTimeResults": [
                        {
                            "timeSet": r.get("timeSet"),
                            "harmonicAverageSpeed": r.get("harmonicAverageSpeed"),
                            "medianSpeed": r.get("medianSpeed"),
                            "averageSpeed": r.get("averageSpeed"),
                            "sampleSize": r.get("sampleSize"),
                            "averageTravelTime": r.get("averageTravelTime"),
                            "travelTimeRatio": r.get("travelTimeRatio"),
                            "standardDeviationSpeed": r.get("standardDeviationSpeed"),
                        }
                        for r in seg.get("segmentTimeResults", [])
                    ],
                }
                all_segments.append(clean_seg)

        except Exception as e:
            print(f"  [WARN] Bỏ qua {fname}: {e}")

    print(f"✓ Loaded {len(all_segments)} unique segments từ {len(json_files)} files")
    return all_segments


def build_html(segments: list) -> str:
    """Tạo HTML hoàn chỉnh với data nhúng sẵn."""

    data_json = json.dumps(segments, ensure_ascii=False, separators=(",", ":"))
    data_size_kb = len(data_json.encode("utf-8")) / 1024
    print(f"✓ Data size: {data_size_kb:.1f} KB")

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HCM Traffic Intelligence — FRC5</title>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<style>
  :root {{
    --bg: #0a0c10;
    --panel: #0f1219;
    --border: #1e2535;
    --accent: #00e5ff;
    --text: #e2e8f0;
    --muted: #64748b;
    --speed-free: #00ff88;
    --speed-mod: #ffd600;
    --speed-slow: #ff6d00;
    --speed-jam: #ff1744;
    --font-mono: 'JetBrains Mono', monospace;
    --font-display: 'Syne', sans-serif;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:var(--font-mono); height:100vh; overflow:hidden; display:flex; flex-direction:column; }}

  /* HEADER */
  header {{ display:flex; align-items:center; justify-content:space-between; padding:0 20px; height:52px; background:var(--panel); border-bottom:1px solid var(--border); flex-shrink:0; z-index:1000; }}
  .logo {{ font-family:var(--font-display); font-weight:800; font-size:15px; color:#fff; display:flex; align-items:center; gap:8px; }}
  .logo-dot {{ width:8px; height:8px; background:var(--accent); border-radius:50%; animation:pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.4;transform:scale(1.5)}} }}
  .header-stats {{ display:flex; gap:24px; font-size:11px; color:var(--muted); }}
  .hstat span {{ color:var(--text); font-weight:600; }}

  /* LAYOUT */
  .layout {{ display:flex; flex:1; overflow:hidden; }}

  /* SIDEBAR */
  .sidebar {{ width:280px; background:var(--panel); border-right:1px solid var(--border); display:flex; flex-direction:column; flex-shrink:0; overflow-y:auto; scrollbar-width:thin; scrollbar-color:var(--border) transparent; }}
  .sidebar::-webkit-scrollbar {{ width:4px; }}
  .sidebar::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:2px; }}

  .section {{ padding:16px; border-bottom:1px solid var(--border); }}
  .section-title {{ font-size:9px; letter-spacing:2px; text-transform:uppercase; color:var(--muted); margin-bottom:12px; font-family:var(--font-display); }}

  /* PERIOD TOGGLE */
  .period-toggle {{ display:flex; background:var(--bg); border-radius:6px; padding:3px; gap:3px; margin-bottom:12px; }}
  .period-btn {{ flex:1; background:transparent; border:none; color:var(--muted); font-family:var(--font-mono); font-size:10px; padding:5px; border-radius:4px; cursor:pointer; transition:all .15s; }}
  .period-btn.active {{ background:var(--border); color:var(--text); }}

  /* TIME GRID */
  .time-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:4px; }}
  .time-btn {{ background:transparent; border:1px solid var(--border); color:var(--muted); font-family:var(--font-mono); font-size:10px; padding:6px 4px; border-radius:4px; cursor:pointer; transition:all .15s; text-align:center; }}
  .time-btn:hover {{ border-color:var(--accent); color:var(--accent); }}
  .time-btn.active {{ background:var(--accent); border-color:var(--accent); color:#000; font-weight:600; }}

  /* METRIC */
  .metric-select {{ width:100%; background:var(--bg); border:1px solid var(--border); color:var(--text); font-family:var(--font-mono); font-size:11px; padding:8px 10px; border-radius:6px; cursor:pointer; outline:none; }}
  .metric-select:focus {{ border-color:var(--accent); }}

  /* LEGEND */
  .legend-items {{ display:flex; flex-direction:column; gap:8px; }}
  .legend-item {{ display:flex; align-items:center; gap:10px; font-size:11px; }}
  .legend-swatch {{ width:28px; height:4px; border-radius:2px; flex-shrink:0; }}
  .legend-range {{ color:var(--muted); font-size:10px; margin-left:auto; }}

  /* STATS */
  .stat-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
  .stat-card {{ background:var(--bg); border:1px solid var(--border); border-radius:6px; padding:10px; }}
  .stat-card .val {{ font-family:var(--font-display); font-size:18px; font-weight:700; color:var(--accent); line-height:1; }}
  .stat-card .lbl {{ font-size:9px; color:var(--muted); margin-top:4px; letter-spacing:1px; text-transform:uppercase; }}

  /* SEGMENT DETAIL */
  #seg-info {{ font-size:10px; line-height:1.8; color:var(--muted); display:none; }}
  #seg-info.visible {{ display:block; }}
  #seg-info .seg-name {{ color:var(--text); font-size:12px; font-weight:600; margin-bottom:8px; line-height:1.4; }}
  #seg-info .seg-row {{ display:flex; justify-content:space-between; border-bottom:1px solid #1a2030; padding:2px 0; }}
  #seg-info .seg-val {{ color:var(--accent); font-weight:600; }}
  .empty-hint {{ font-size:10px; color:var(--muted); font-style:italic; }}

  /* FILTER */
  .filter-row {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; font-size:10px; color:var(--muted); }}
  .filter-row input[type=range] {{ width:140px; accent-color:var(--accent); }}
  .filter-val {{ color:var(--text); font-weight:600; min-width:40px; text-align:right; }}

  /* MAP */
  #map {{ flex:1; background:#08090d; }}

  /* LOADING */
  #loading {{ position:fixed; inset:0; background:var(--bg); display:flex; flex-direction:column; align-items:center; justify-content:center; z-index:9999; gap:16px; transition:opacity .5s; }}
  #loading.hidden {{ opacity:0; pointer-events:none; }}
  .load-title {{ font-family:var(--font-display); font-size:28px; font-weight:800; }}
  .load-bar-wrap {{ width:320px; height:2px; background:var(--border); border-radius:1px; }}
  .load-bar {{ height:100%; background:var(--accent); border-radius:1px; width:0%; transition:width .2s; }}
  .load-msg {{ font-size:11px; color:var(--muted); letter-spacing:1px; }}
  .load-sub {{ font-size:10px; color:#334155; }}

  /* TOOLTIP */
  .leaflet-tooltip-pane .traffic-tip {{ background:var(--panel) !important; border:1px solid var(--border) !important; color:var(--text) !important; font-family:var(--font-mono) !important; font-size:11px !important; border-radius:6px !important; padding:8px 12px !important; box-shadow:0 8px 24px rgba(0,0,0,.5) !important; white-space:nowrap !important; }}
  .leaflet-control-zoom a {{ background:var(--panel) !important; color:var(--text) !important; border-color:var(--border) !important; }}

  /* SPEED ANIMATION for congested roads */
  @keyframes congestion-pulse {{
    0%,100% {{ opacity: 0.85; }}
    50% {{ opacity: 0.4; }}
  }}
</style>
</head>
<body>

<!-- LOADING OVERLAY -->
<div id="loading">
  <div class="load-title">HCM Traffic Intelligence</div>
  <div style="font-size:11px;color:var(--muted);letter-spacing:3px;">FRC5 · TOMTOM MOVE</div>
  <div class="load-bar-wrap"><div class="load-bar" id="load-bar"></div></div>
  <div class="load-msg" id="load-msg">Đang tải dữ liệu...</div>
  <div class="load-sub" id="load-sub"></div>
</div>

<!-- HEADER -->
<header>
  <div class="logo">
    <div class="logo-dot"></div>
    HCM Traffic Intelligence
    <span style="font-size:10px;color:var(--muted);font-weight:400;margin-left:4px;">FRC5 · 2024-08-01</span>
  </div>
  <div class="header-stats">
    <div class="hstat">Segments: <span id="h-segs">—</span></div>
    <div class="hstat">Time Slots: <span id="h-slots">—</span></div>
    <div class="hstat">Nguồn: <span>TomTom Move</span></div>
  </div>
</header>

<!-- MAIN LAYOUT -->
<div class="layout">

  <!-- SIDEBAR -->
  <div class="sidebar">

    <!-- METRIC -->
    <div class="section">
      <div class="section-title">Chỉ số tốc độ</div>
      <select class="metric-select" id="metric-sel">
        <option value="harmonicAverageSpeed">Harmonic Average Speed</option>
        <option value="medianSpeed">Median Speed</option>
        <option value="averageSpeed">Average Speed</option>
      </select>
    </div>

    <!-- TIME PERIOD & SLOTS -->
    <div class="section">
      <div class="section-title">Khung giờ</div>
      <div class="period-toggle">
        <button class="period-btn active" data-period="morning">Sáng 07–10h</button>
        <button class="period-btn" data-period="afternoon">Chiều 15–18h</button>
      </div>
      <div class="time-grid" id="time-grid"></div>
    </div>

    <!-- SPEED FILTER -->
    <div class="section">
      <div class="section-title">Lọc theo tốc độ</div>
      <div class="filter-row">
        <span>Tốc độ tối đa hiển thị</span>
        <span class="filter-val" id="filter-val">— km/h</span>
      </div>
      <input type="range" id="speed-filter" min="0" max="80" value="80" step="5"
             style="width:100%;accent-color:var(--accent);margin-bottom:8px;">
      <div class="filter-row" style="margin-bottom:0">
        <span style="font-size:9px;color:#334155;">Chỉ hiện đường ≤ ngưỡng này</span>
        <button onclick="resetFilter()" style="background:transparent;border:1px solid var(--border);color:var(--muted);font-family:var(--font-mono);font-size:9px;padding:2px 6px;border-radius:3px;cursor:pointer;">Reset</button>
      </div>
    </div>

    <!-- LEGEND -->
    <div class="section">
      <div class="section-title">Chú thích màu (km/h)</div>
      <div class="legend-items">
        <div class="legend-item">
          <div class="legend-swatch" style="background:var(--speed-free)"></div>
          <div>Thông thoáng</div>
          <div class="legend-range">&gt; 30</div>
        </div>
        <div class="legend-item">
          <div class="legend-swatch" style="background:var(--speed-mod)"></div>
          <div>Bình thường</div>
          <div class="legend-range">20 – 30</div>
        </div>
        <div class="legend-item">
          <div class="legend-swatch" style="background:var(--speed-slow)"></div>
          <div>Chậm</div>
          <div class="legend-range">10 – 20</div>
        </div>
        <div class="legend-item">
          <div class="legend-swatch" style="background:var(--speed-jam)"></div>
          <div>Kẹt xe</div>
          <div class="legend-range">&lt; 10</div>
        </div>
        <div class="legend-item">
          <div class="legend-swatch" style="background:#334155"></div>
          <div>Không có dữ liệu</div>
          <div class="legend-range">—</div>
        </div>
      </div>
    </div>

    <!-- NETWORK STATS -->
    <div class="section">
      <div class="section-title">Thống kê mạng lưới</div>
      <div class="stat-grid">
        <div class="stat-card"><div class="val" id="s-avg">—</div><div class="lbl">TB tốc độ</div></div>
        <div class="stat-card"><div class="val" id="s-jam">—</div><div class="lbl">Kẹt xe</div></div>
        <div class="stat-card"><div class="val" id="s-free">—</div><div class="lbl">Thông thoáng</div></div>
        <div class="stat-card"><div class="val" id="s-visible">—</div><div class="lbl">Đang hiển thị</div></div>
      </div>
    </div>

    <!-- SEGMENT DETAIL -->
    <div class="section" style="flex:1">
      <div class="section-title">Chi tiết đoạn đường</div>
      <div id="seg-info">
        <div class="seg-name" id="si-name"></div>
        <div class="seg-row"><span>Tốc độ</span><span class="seg-val" id="si-speed"></span></div>
        <div class="seg-row"><span>Giới hạn tốc độ</span><span class="seg-val" id="si-limit"></span></div>
        <div class="seg-row"><span>TG di chuyển TB</span><span class="seg-val" id="si-tt"></span></div>
        <div class="seg-row"><span>Travel Time Ratio</span><span class="seg-val" id="si-ttr"></span></div>
        <div class="seg-row"><span>Std Dev Speed</span><span class="seg-val" id="si-std"></span></div>
        <div class="seg-row"><span>Sample Size</span><span class="seg-val" id="si-sample"></span></div>
        <div class="seg-row"><span>Chiều dài</span><span class="seg-val" id="si-dist"></span></div>
        <div class="seg-row"><span>FRC</span><span class="seg-val" id="si-frc"></span></div>
        <div class="seg-row"><span>Segment ID</span><span class="seg-val" id="si-id" style="font-size:9px;"></span></div>
      </div>
      <div class="empty-hint" id="si-hint">← Click một đoạn đường trên bản đồ</div>
    </div>

  </div><!-- /sidebar -->

  <div id="map"></div>
</div>

<!-- ═══════════════════════════════════════════════════════
     EMBEDDED DATA — injected by build_traffic_map.py
     {len(segments)} segments · {data_size_kb:.1f} KB
     ═══════════════════════════════════════════════════════ -->
<script>
const ALL_SEGMENTS = {data_json};
</script>

<script>
// ══════════════════════════════════════════════
//  TIME SLOT CONFIG (matches TomTom timeSet IDs)
// ══════════════════════════════════════════════
const TIME_SLOTS = {{
  2:'07:00', 3:'07:15', 4:'07:30', 5:'07:45',
  6:'08:00', 7:'08:15', 8:'08:30', 9:'08:45',
  10:'09:00',11:'09:15',12:'09:30',13:'09:45',
  14:'15:00',15:'15:15',16:'15:30',17:'15:45',
  18:'16:00',19:'16:15',20:'16:30',21:'16:45',
  22:'17:00',23:'17:15',24:'17:30',25:'17:45'
}};
const MORNING_SETS   = [2,3,4,5,6,7,8,9,10,11,12,13];
const AFTERNOON_SETS = [14,15,16,17,18,19,20,21,22,23,24,25];

// ── STATE ──────────────────────────────────────
let currentTimeSet = 6;      // 08:00 peak
let currentMetric  = 'harmonicAverageSpeed';
let currentPeriod  = 'morning';
let speedFilterMax = 80;
let polylines = [];
let map;

// ── COLOR HELPERS ──────────────────────────────
function speedToColor(speed) {{
  if (speed === null || speed === undefined) return '#334155';
  if (speed >= 30) return '#00ff88';
  if (speed >= 20) return '#ffd600';
  if (speed >= 10) return '#ff6d00';
  return '#ff1744';
}}
function speedToWeight(speed, zoom) {{
  const base = (speed !== null && speed < 10) ? 5
             : (speed !== null && speed < 20) ? 4 : 3;
  return Math.max(2, base + (zoom - 14) * 0.4);
}}
function getSegResult(seg, ts) {{
  return seg.segmentTimeResults.find(r => r.timeSet === ts) || null;
}}
function getSpeed(seg, ts, metric) {{
  const r = getSegResult(seg, ts);
  return r ? r[metric] : null;
}}

// ── CANVAS RENDERER (performance key) ──────────
const canvasRenderer = L.canvas({{ padding: 0.5 }});

// ══════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════
async function init() {{
  setLoad(10, 'Khởi tạo bản đồ...', '');

  // Leaflet map — preferCanvas để render hàng nghìn segment không giật
  map = L.map('map', {{
    center: [10.776, 106.700],
    zoom: 14,
    zoomControl: true,
    preferCanvas: true,
  }});

  // Dark CartoDB tile
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '© OpenStreetMap contributors © CARTO',
    subdomains: 'abcd',
    maxZoom: 19
  }}).addTo(map);

  setLoad(30, 'Xử lý dữ liệu TomTom...', ALL_SEGMENTS.length + ' segments');
  await sleep(20);

  // Header stats
  document.getElementById('h-segs').textContent  = ALL_SEGMENTS.length.toLocaleString();
  document.getElementById('h-slots').textContent = Object.keys(TIME_SLOTS).length;

  // Auto-center on actual data bounds
  fitMapToBounds();

  setLoad(60, 'Vẽ đường giao thông...', 'Canvas renderer');
  await sleep(20);

  buildTimeBtns();
  renderAllPolylines();
  updateStats();
  setupFilter();
  setupEvents();

  setLoad(100, 'Hoàn tất!', '');
  await sleep(400);
  document.getElementById('loading').classList.add('hidden');
}}

function sleep(ms) {{ return new Promise(r => setTimeout(r, ms)); }}
function setLoad(pct, msg, sub) {{
  document.getElementById('load-bar').style.width = pct + '%';
  document.getElementById('load-msg').textContent = msg;
  document.getElementById('load-sub').textContent = sub || '';
}}

// ── FIT BOUNDS ─────────────────────────────────
function fitMapToBounds() {{
  const lats = [], lngs = [];
  ALL_SEGMENTS.forEach(seg => {{
    seg.shape.forEach(p => {{ lats.push(p.latitude); lngs.push(p.longitude); }});
  }});
  if (!lats.length) return;
  const bounds = [
    [Math.min(...lats), Math.min(...lngs)],
    [Math.max(...lats), Math.max(...lngs)]
  ];
  map.fitBounds(bounds, {{ padding: [40, 40] }});
}}

// ══════════════════════════════════════════════
//  BUILD TIME BUTTONS
// ══════════════════════════════════════════════
function buildTimeBtns() {{
  const grid  = document.getElementById('time-grid');
  grid.innerHTML = '';
  const slots = currentPeriod === 'morning' ? MORNING_SETS : AFTERNOON_SETS;
  slots.forEach(ts => {{
    const btn = document.createElement('button');
    btn.className = 'time-btn' + (ts === currentTimeSet ? ' active' : '');
    btn.textContent = TIME_SLOTS[ts];
    btn.addEventListener('click', () => {{
      document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentTimeSet = ts;
      updateAllColors();
      updateStats();
    }});
    grid.appendChild(btn);
  }});
}}

// ══════════════════════════════════════════════
//  RENDER POLYLINES (one-time, then only recolor)
// ══════════════════════════════════════════════
function renderAllPolylines() {{
  polylines.forEach(p => p.remove());
  polylines = [];

  const zoom = map.getZoom();

  ALL_SEGMENTS.forEach(seg => {{
    if (!seg.shape || seg.shape.length < 2) return;

    const latlngs = seg.shape.map(p => [p.latitude, p.longitude]);
    const speed   = getSpeed(seg, currentTimeSet, currentMetric);
    const color   = speedToColor(speed);
    const weight  = speedToWeight(speed, zoom);
    const visible = speed === null || speed <= speedFilterMax;

    const pl = L.polyline(latlngs, {{
      color,
      weight,
      opacity: visible ? 0.85 : 0,
      renderer: canvasRenderer,
      lineCap: 'round',
      lineJoin: 'round',
    }}).addTo(map);

    // Tooltip — lazy render chỉ khi hover
    pl.bindTooltip('', {{ className: 'traffic-tip', sticky: true, direction: 'top', offset: [0, -4] }});

    pl.on('mouseover', function(e) {{
      const r = getSegResult(seg, currentTimeSet);
      const spd = r ? r[currentMetric] : null;
      const spdStr = spd !== null ? spd.toFixed(1) + ' km/h' : '—';
      const col = speedToColor(spd);
      pl.setTooltipContent(
        `<b>${{seg.streetName}}</b><br>` +
        `Tốc độ: <b style="color:${{col}}">${{spdStr}}</b> / ${{seg.speedLimit}} km/h<br>` +
        `Chiều dài: ${{seg.distance ? seg.distance.toFixed(0) : '—'}} m · FRC ${{seg.frc}}`
      );
      pl.openTooltip(e.latlng);
      if (visible) pl.setStyle({{ opacity: 1, weight: weight + 2 }});
    }});

    pl.on('mouseout', function() {{
      if (visible) pl.setStyle({{ opacity: 0.85, weight }});
    }});

    pl.on('click', function() {{ showSegDetail(seg); }});

    pl._seg    = seg;
    pl._weight = weight;
    polylines.push(pl);
  }});
}}

// ══════════════════════════════════════════════
//  UPDATE COLORS ONLY (no DOM rebuild)
// ══════════════════════════════════════════════
function updateAllColors() {{
  const zoom = map.getZoom();
  polylines.forEach(pl => {{
    const seg     = pl._seg;
    const speed   = getSpeed(seg, currentTimeSet, currentMetric);
    const color   = speedToColor(speed);
    const weight  = speedToWeight(speed, zoom);
    const visible = speed === null || speed <= speedFilterMax;
    pl._weight = weight;
    pl.setStyle({{ color, weight, opacity: visible ? 0.85 : 0 }});
  }});
}}

// ══════════════════════════════════════════════
//  SPEED FILTER
// ══════════════════════════════════════════════
function setupFilter() {{
  const slider = document.getElementById('speed-filter');
  const val    = document.getElementById('filter-val');
  val.textContent = slider.value + ' km/h';

  slider.addEventListener('input', () => {{
    speedFilterMax = parseInt(slider.value);
    val.textContent = speedFilterMax === 80 ? 'Tất cả' : speedFilterMax + ' km/h';
    applyFilter();
    updateStats();
  }});
}}

function applyFilter() {{
  polylines.forEach(pl => {{
    const speed = getSpeed(pl._seg, currentTimeSet, currentMetric);
    const show  = speed === null || speed <= speedFilterMax;
    pl.setStyle({{ opacity: show ? 0.85 : 0 }});
  }});
}}

function resetFilter() {{
  document.getElementById('speed-filter').value = 80;
  speedFilterMax = 80;
  document.getElementById('filter-val').textContent = 'Tất cả';
  applyFilter();
  updateStats();
}}

// ══════════════════════════════════════════════
//  STATS
// ══════════════════════════════════════════════
function updateStats() {{
  let speeds = [], jam = 0, free = 0, visible = 0;

  ALL_SEGMENTS.forEach(seg => {{
    const r = getSegResult(seg, currentTimeSet);
    if (!r) return;
    const s = r[currentMetric];
    if (s === null || s === undefined) return;

    const show = s <= speedFilterMax;
    if (show) {{
      speeds.push(s);
      visible++;
      if (s < 10)  jam++;
      if (s >= 30) free++;
    }}
  }});

  const avg    = speeds.length ? (speeds.reduce((a,b)=>a+b,0)/speeds.length).toFixed(1) + '' : '—';
  const jamPct = speeds.length ? ((jam/speeds.length)*100).toFixed(0) + '%' : '—';
  const freePct= speeds.length ? ((free/speeds.length)*100).toFixed(0)+ '%' : '—';

  document.getElementById('s-avg').textContent     = avg + (speeds.length ? '' : '');
  document.getElementById('s-jam').textContent     = jamPct;
  document.getElementById('s-free').textContent    = freePct;
  document.getElementById('s-visible').textContent = visible.toLocaleString();
}}

// ══════════════════════════════════════════════
//  SEGMENT DETAIL PANEL
// ══════════════════════════════════════════════
function showSegDetail(seg) {{
  const r = getSegResult(seg, currentTimeSet);
  const spd = r ? r[currentMetric] : null;
  document.getElementById('si-name').textContent    = seg.streetName || 'Unknown';
  document.getElementById('si-speed').textContent   = spd !== null ? spd.toFixed(1) + ' km/h' : '—';
  document.getElementById('si-speed').style.color   = speedToColor(spd);
  document.getElementById('si-limit').textContent   = (seg.speedLimit || '—') + ' km/h';
  document.getElementById('si-tt').textContent      = r?.averageTravelTime  ? r.averageTravelTime.toFixed(1) + ' s' : '—';
  document.getElementById('si-ttr').textContent     = r?.travelTimeRatio    ? r.travelTimeRatio.toFixed(2)          : '—';
  document.getElementById('si-std').textContent     = r?.standardDeviationSpeed ? r.standardDeviationSpeed.toFixed(1) + ' km/h' : '—';
  document.getElementById('si-sample').textContent  = r?.sampleSize ?? '—';
  document.getElementById('si-dist').textContent    = seg.distance ? seg.distance.toFixed(0) + ' m' : '—';
  document.getElementById('si-frc').textContent     = 'FRC ' + (seg.frc ?? '—');
  document.getElementById('si-id').textContent      = seg.segmentId ?? '—';
  document.getElementById('seg-info').classList.add('visible');
  document.getElementById('si-hint').style.display  = 'none';
}}

// ══════════════════════════════════════════════
//  EVENTS
// ══════════════════════════════════════════════
function setupEvents() {{
  // Metric change
  document.getElementById('metric-sel').addEventListener('change', e => {{
    currentMetric = e.target.value;
    updateAllColors();
    updateStats();
  }});

  // Period toggle
  document.querySelectorAll('.period-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentPeriod  = btn.dataset.period;
      currentTimeSet = currentPeriod === 'morning' ? 2 : 14;
      buildTimeBtns();
      updateAllColors();
      updateStats();
    }});
  }});

  // Zoom: reweight lines
  map.on('zoomend', () => updateAllColors());
}}

// ── LAUNCH ─────────────────────────────────────
init();
</script>
</body>
</html>"""
    return html


def main():
    print("=" * 60)
    print("  HCM Traffic Map Builder")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"\n[ERROR] Không tìm thấy thư mục: {{DATA_DIR}}")
        print("  Hãy đặt script này ở thư mục gốc project (cùng cấp ml_core/).")
        return

    print(f"\n📂 Data dir : {{DATA_DIR}}")
    print(f"📄 Output   : {{OUTPUT_HTML}}\n")

    segments = load_all_segments(DATA_DIR)

    if not segments:
        print("[ERROR] Không có segment nào được load.")
        return

    print(f"\n🔨 Building HTML...")
    html = build_html(segments)

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_mb = OUTPUT_HTML.stat().st_size / 1024 / 1024
    print(f"✓ Saved → {{OUTPUT_HTML}}  ({size_mb:.2f} MB)")
    print(f"\n✅ Xong! Mở file sau bằng Chrome hoặc Firefox:")
    print(f"   {OUTPUT_HTML.resolve()}")
    print()


if __name__ == "__main__":
    main()