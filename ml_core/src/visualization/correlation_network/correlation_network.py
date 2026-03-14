#!/usr/bin/env python3
"""
Correlation Network Graph Generator — tối ưu cho ma trận lớn
Đọc CSV streaming (không load hết RAM), chỉ giữ edges vượt ngưỡng.

Cách dùng:
    python correlation_network.py <input.csv> [output.html] [threshold] [max_edges]

Ví dụ:
    python correlation_network.py corr_mean_nxn.csv
    python correlation_network.py corr_mean_nxn.csv graph.html 0.8
    python correlation_network.py corr_mean_nxn.csv graph.html 0.8 50000

Mặc định: threshold=0.7, max_edges=30000
Với 9534 nodes (~45M cặp), dùng threshold >= 0.7 để file HTML chạy được.
"""

import sys
import csv
import json
import os

MAX_EDGES_DEFAULT = 30_000
THRESHOLD_DEFAULT = 0.7


def read_and_filter_csv(filepath, threshold, max_edges):
    """Đọc CSV từng dòng (streaming), chỉ giữ edges vượt ngưỡng. Tiết kiệm RAM."""
    edges = []
    seen = set()
    was_capped = False

    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        headers = next(reader)
        col_ids = [h.strip() for h in headers[1:]]
        total_cols = len(col_ids)

        print(f"   Số nodes (cột): {total_cols:,}")
        print(f"   Ngưỡng |r| >= {threshold}  |  Giới hạn edges: {max_edges:,}")
        print(f"   Đang quét...", flush=True)

        for row_idx, row in enumerate(reader):
            if not row:
                continue
            row_id = row[0].strip()

            if row_idx % 200 == 0 and row_idx > 0:
                pct = row_idx / total_cols * 100
                print(f"   Dòng {row_idx:,}/{total_cols:,} ({pct:.0f}%)  edges: {len(edges):,}",
                      end='\r', flush=True)

            if was_capped:
                continue  # đã đủ edges, vẫn đọc tiếp để không lỗi EOF nhưng bỏ qua

            for j, col_id in enumerate(col_ids):
                if row_id == col_id:
                    continue
                key = (min(row_id, col_id), max(row_id, col_id))
                if key in seen:
                    continue
                try:
                    val = float(row[j + 1])
                except (ValueError, IndexError):
                    continue
                if abs(val) >= threshold:
                    seen.add(key)
                    edges.append({"source": row_id, "target": col_id, "value": round(val, 4)})
                    if len(edges) >= max_edges:
                        was_capped = True
                        break

    print()

    active_nodes = set()
    for e in edges:
        active_nodes.add(e["source"])
        active_nodes.add(e["target"])

    node_ids = sorted(active_nodes)
    return node_ids, edges, was_capped


def generate_html(node_ids, edges, threshold, output_path):
    nodes_json = json.dumps(
        [{"id": n, "label": n} for n in node_ids], ensure_ascii=False
    )
    edges_json = json.dumps(edges, ensure_ascii=False)
    threshold_str = f"{threshold:.2f}"
    node_count = len(node_ids)
    edge_count = len(edges)

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Correlation Network Graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #f8f8f7;
  color: #2c2c2a;
  min-height: 100vh;
}}
header {{
  background: #fff;
  border-bottom: 1px solid #e0ddd5;
  padding: 12px 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}}
header h1 {{ font-size: 15px; font-weight: 500; flex: 1; min-width: 160px; }}
.stats {{ display: flex; gap: 14px; font-size: 13px; color: #888780; }}
.stats span b {{ color: #2c2c2a; font-weight: 500; }}
.controls {{ display: flex; align-items: center; gap: 10px; font-size: 13px; flex-wrap: wrap; }}
label {{ color: #888780; white-space: nowrap; }}
input[type=range] {{ width: 110px; accent-color: #534AB7; cursor: pointer; }}
#threshold-val {{ font-weight: 500; min-width: 34px; text-align: right; font-size: 13px; }}
select {{
  font-size: 13px; padding: 4px 8px;
  border: 1px solid #d3d1c7; border-radius: 6px;
  background: #fff; color: #2c2c2a; cursor: pointer;
}}
#graph-container {{
  width: 100%; height: calc(100vh - 56px);
  position: relative; overflow: hidden;
}}
svg {{ width: 100%; height: 100%; }}
.link {{ stroke-linecap: round; transition: opacity 0.15s; }}
.node circle {{ transition: r 0.15s; cursor: pointer; }}
.node text {{ pointer-events: none; user-select: none; }}
#tooltip {{
  position: absolute;
  background: #fff;
  border: 1px solid #d3d1c7;
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 13px;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.12s;
  max-width: 240px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  z-index: 10;
}}
.tt-title {{ font-weight: 500; font-size: 14px; margin-bottom: 6px; }}
.tt-row {{ display: flex; justify-content: space-between; gap: 12px; color: #5f5e5a; margin: 2px 0; font-size: 12px; }}
.tt-row b {{ color: #2c2c2a; font-weight: 500; }}
.legend {{
  position: absolute; bottom: 18px; left: 18px;
  background: rgba(255,255,255,0.92);
  border: 1px solid #e0ddd5;
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 12px;
}}
.legend-title {{ font-weight: 500; margin-bottom: 8px; font-size: 13px; }}
.legend-item {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; color: #5f5e5a; }}
.legend-line {{ height: 3px; width: 30px; border-radius: 2px; }}
.hint {{
  position: absolute; bottom: 18px; right: 18px;
  font-size: 11px; color: #b4b2a9; text-align: right; line-height: 1.6;
}}
#warning {{
  display: none;
  position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
  background: #FAEEDA; border: 1px solid #FAC775;
  color: #633806; border-radius: 8px; padding: 8px 16px;
  font-size: 13px; z-index: 20; white-space: nowrap;
}}
</style>
</head>
<body>
<header>
  <h1>Correlation Network Graph</h1>
  <div class="stats">
    <span>Nodes: <b id="stat-nodes">{node_count}</b></span>
    <span>Edges: <b id="stat-edges">{edge_count}</b></span>
  </div>
  <div class="controls">
    <label>Ngưỡng |r| &ge;</label>
    <input type="range" id="threshold-slider" min="0" max="1" step="0.01" value="{threshold_str}">
    <span id="threshold-val">{threshold_str}</span>
    <label style="margin-left:8px">Layout</label>
    <select id="layout-select">
      <option value="force">Force</option>
      <option value="radial">Radial</option>
    </select>
    <label style="margin-left:8px">Labels</label>
    <select id="label-select">
      <option value="auto">Tự động</option>
      <option value="on">Luôn hiện</option>
      <option value="off">Ẩn</option>
    </select>
  </div>
</header>

<div id="graph-container">
  <svg id="svg"></svg>
  <div id="tooltip"></div>
  <div id="warning">⚠ Quá nhiều nodes để hiển thị label — hãy tăng ngưỡng hoặc chọn "Luôn hiện"</div>
  <div class="legend">
    <div class="legend-title">Hệ số tương quan</div>
    <div class="legend-item"><div class="legend-line" style="background:#1D9E75;height:4px"></div>Dương mạnh (&ge;0.7)</div>
    <div class="legend-item"><div class="legend-line" style="background:#5DCAA5;height:2.5px"></div>Dương vừa (0.3–0.7)</div>
    <div class="legend-item"><div class="legend-line" style="background:#9FE1CB;height:1.5px"></div>Dương yếu (&lt;0.3)</div>
    <div class="legend-item"><div class="legend-line" style="background:#D85A30;height:4px"></div>Âm mạnh (&le;−0.7)</div>
    <div class="legend-item"><div class="legend-line" style="background:#F0997B;height:2px"></div>Âm yếu (−0.7–0)</div>
  </div>
  <div class="hint">Scroll để zoom · Kéo để pan<br>Click node để highlight · Drag node để di chuyển</div>
</div>

<script>
const ALL_NODES = {nodes_json};
const ALL_EDGES = {edges_json};
const INIT_THRESHOLD = {threshold_str};
const MAX_LABEL_NODES = 300;

let activeThreshold = INIT_THRESHOLD;
let selectedNode = null;

const svg = d3.select("#svg");
const container = document.getElementById("graph-container");
const tooltip = document.getElementById("tooltip");
let width = container.clientWidth;
let height = container.clientHeight;

const g = svg.append("g");
const zoom = d3.zoom().scaleExtent([0.05, 10])
  .on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom).on("dblclick.zoom", null);

function linkColor(v) {{
  if (v >= 0.7)  return "#1D9E75";
  if (v >= 0.3)  return "#5DCAA5";
  if (v >= 0)    return "#9FE1CB";
  if (v >= -0.3) return "#F0997B";
  if (v >= -0.7) return "#F09595";
  return "#D85A30";
}}
function linkWidth(v) {{
  const a = Math.abs(v);
  return a >= 0.7 ? 2.5 : a >= 0.3 ? 1.5 : 1;
}}

let simulation, linkSel, nodeSel;

function getLabelMode(nodeCount) {{
  const mode = document.getElementById("label-select").value;
  if (mode === "on")  return true;
  if (mode === "off") return false;
  return nodeCount <= MAX_LABEL_NODES;
}}

function buildGraph(threshold) {{
  g.selectAll("*").remove();
  simulation && simulation.stop();
  selectedNode = null;

  const edgeData = ALL_EDGES.filter(e => Math.abs(e.value) >= threshold);
  const nodeSet = new Set();
  edgeData.forEach(e => {{ nodeSet.add(e.source); nodeSet.add(e.target); }});
  const nodeData = ALL_NODES
    .filter(n => nodeSet.has(n.id))
    .map(d => ({{...d}}));
  const nodeMap = {{}};
  nodeData.forEach(n => nodeMap[n.id] = n);

  const edges = edgeData.map(e => ({{
    ...e,
    source: nodeMap[e.source] || e.source,
    target: nodeMap[e.target] || e.target
  }}));

  const degree = {{}};
  nodeData.forEach(n => degree[n.id] = 0);
  edges.forEach(e => {{
    const sid = typeof e.source === 'object' ? e.source.id : e.source;
    const tid = typeof e.target === 'object' ? e.target.id : e.target;
    degree[sid] = (degree[sid] || 0) + 1;
    degree[tid] = (degree[tid] || 0) + 1;
  }});
  nodeData.forEach(n => n.degree = degree[n.id] || 0);

  document.getElementById("stat-nodes").textContent = nodeData.length.toLocaleString();
  document.getElementById("stat-edges").textContent = edges.length.toLocaleString();

  const warning = document.getElementById("warning");
  warning.style.display = nodeData.length > MAX_LABEL_NODES && getLabelMode(nodeData.length) ? 'block' : 'none';
  setTimeout(() => warning.style.display = 'none', 3000);

  const showLabels = getLabelMode(nodeData.length);
  const maxDeg = Math.max(...nodeData.map(n => n.degree), 1);

  linkSel = g.append("g").selectAll("line").data(edges).join("line")
    .attr("stroke", d => linkColor(d.value))
    .attr("stroke-width", d => linkWidth(d.value))
    .attr("opacity", 0.6);

  nodeSel = g.append("g").selectAll("g").data(nodeData).join("g")
    .style("cursor", "pointer")
    .call(d3.drag()
      .on("start", (ev, d) => {{ if (simulation && !ev.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
      .on("drag",  (ev, d) => {{ d.fx=ev.x; d.fy=ev.y; }})
      .on("end",   (ev, d) => {{
        if (simulation && !ev.active) simulation.alphaTarget(0);
        if (document.getElementById("layout-select").value !== "radial") {{ d.fx=null; d.fy=null; }}
      }}))
    .on("click", (ev, d) => {{
      ev.stopPropagation();
      selectedNode = selectedNode === d.id ? null : d.id;
      highlight(selectedNode);
    }})
    .on("mouseover", onOver)
    .on("mousemove", onMove)
    .on("mouseout",  onOut);

  nodeSel.append("circle")
    .attr("r", d => 5 + (d.degree / maxDeg) * 12)
    .attr("fill", "#7F77DD")
    .attr("fill-opacity", 0.82)
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5);

  if (showLabels) {{
    nodeSel.append("text")
      .text(d => d.label)
      .attr("text-anchor", "middle")
      .attr("dy", d => -(6 + (d.degree / maxDeg) * 12) - 4)
      .attr("font-size", "10px")
      .attr("fill", "#444441");
  }}

  const layout = document.getElementById("layout-select").value;
  if (layout === "radial") {{
    const n = nodeData.length;
    const r = Math.min(width, height) * 0.38;
    nodeData.forEach((nd, i) => {{
      const angle = (2 * Math.PI * i) / n - Math.PI / 2;
      nd.x = width/2 + r * Math.cos(angle);
      nd.y = height/2 + r * Math.sin(angle);
      nd.fx = nd.x; nd.fy = nd.y;
    }});
    tick();
  }} else {{
    nodeData.forEach(nd => {{ nd.fx = null; nd.fy = null; }});
    simulation = d3.forceSimulation(nodeData)
      .force("link", d3.forceLink(edges).id(d => d.id)
        .distance(d => 60 + (1 - Math.abs(d.value)) * 80))
      .force("charge", d3.forceManyBody()
        .strength(nodeData.length > 500 ? -30 : -100))
      .force("center", d3.forceCenter(width/2, height/2))
      .force("collision", d3.forceCollide().radius(d => 8 + (d.degree/maxDeg)*14))
      .alphaDecay(nodeData.length > 1000 ? 0.05 : 0.028)
      .on("tick", tick);
  }}
}}

function tick() {{
  if (linkSel) linkSel
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  if (nodeSel) nodeSel.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}}

function highlight(nid) {{
  if (!nodeSel || !linkSel) return;
  if (!nid) {{
    nodeSel.selectAll("circle").attr("opacity", 1);
    linkSel.attr("opacity", 0.6);
    return;
  }}
  const conn = new Set([nid]);
  linkSel.each(d => {{
    const s = typeof d.source==='object'?d.source.id:d.source;
    const t = typeof d.target==='object'?d.target.id:d.target;
    if (s===nid||t===nid) {{ conn.add(s); conn.add(t); }}
  }});
  nodeSel.selectAll("circle").attr("opacity", d => conn.has(d.id) ? 1 : 0.12);
  linkSel.attr("opacity", d => {{
    const s = typeof d.source==='object'?d.source.id:d.source;
    const t = typeof d.target==='object'?d.target.id:d.target;
    return (s===nid||t===nid) ? 0.9 : 0.04;
  }});
}}

svg.on("click", () => {{ selectedNode=null; highlight(null); }});

function onOver(ev, d) {{
  const top5 = ALL_EDGES
    .filter(e => e.source===d.id||e.target===d.id)
    .sort((a,b) => Math.abs(b.value)-Math.abs(a.value))
    .slice(0, 6)
    .map(e => {{
      const other = e.source===d.id?e.target:e.source;
      const sign = e.value>=0?'+':'';
      return `<div class="tt-row"><span>${{other}}</span><b>${{sign}}${{e.value.toFixed(3)}}</b></div>`;
    }}).join('');
  tooltip.innerHTML = `
    <div class="tt-title">${{d.label}}</div>
    <div class="tt-row"><span>Kết nối</span><b>${{d.degree}}</b></div>
    ${{top5 ? '<hr style="border:none;border-top:1px solid #e0ddd5;margin:6px 0">' + top5 : ''}}
  `;
  tooltip.style.opacity = 1;
}}
function onMove(ev) {{
  const rect = container.getBoundingClientRect();
  let x = ev.clientX - rect.left + 14;
  let y = ev.clientY - rect.top - 10;
  if (x + 250 > width) x -= 264;
  tooltip.style.left = x + "px";
  tooltip.style.top  = y + "px";
}}
function onOut() {{ tooltip.style.opacity = 0; }}

document.getElementById("threshold-slider").addEventListener("input", function() {{
  const v = parseFloat(this.value).toFixed(2);
  document.getElementById("threshold-val").textContent = v;
  activeThreshold = parseFloat(v);
  buildGraph(activeThreshold);
}});
document.getElementById("layout-select").addEventListener("change", () => buildGraph(activeThreshold));
document.getElementById("label-select").addEventListener("change", () => buildGraph(activeThreshold));

window.addEventListener("resize", () => {{
  width = container.clientWidth; height = container.clientHeight;
  if (simulation) simulation.force("center", d3.forceCenter(width/2, height/2)).alpha(0.3).restart();
}});

buildGraph(activeThreshold);
</script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ Xuất xong: {output_path}")
    print(f"   Nodes hiển thị: {node_count:,}  |  Edges: {edge_count:,}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    input_csv = args[0]
    if not os.path.exists(input_csv):
        print(f"❌ Không tìm thấy file: {input_csv}")
        sys.exit(1)

    base        = os.path.splitext(os.path.basename(input_csv))[0]
    output_html = args[1] if len(args) > 1 else f"{base}_network.html"
    threshold   = float(args[2]) if len(args) > 2 else THRESHOLD_DEFAULT
    max_edges   = int(args[3])   if len(args) > 3 else MAX_EDGES_DEFAULT

    print(f"📂 File: {input_csv}")
    print(f"💡 Tip: Với 9534 nodes (~45M cặp), threshold >= 0.7 giúp file HTML chạy mượt.")

    node_ids, edges, was_capped = read_and_filter_csv(input_csv, threshold, max_edges)

    if was_capped:
        print(f"⚠️  Đã chạm giới hạn {max_edges:,} edges — thử tăng threshold:")
        print(f"   python {os.path.basename(sys.argv[0])} {input_csv} {output_html} {threshold + 0.05:.2f}")

    print(f"   Nodes: {len(node_ids):,}  |  Edges: {len(edges):,}")
    generate_html(node_ids, edges, threshold=threshold, output_path=output_html)
    print(f"👉 Mở '{output_html}' trong trình duyệt.")


if __name__ == "__main__":
    main()