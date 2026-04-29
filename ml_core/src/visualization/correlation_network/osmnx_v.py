import osmnx as ox
import networkx as nx
import folium
import numpy as np
from shapely.geometry import Polygon

# =========================
# 1. LOAD NPZ DATA
# =========================
from pathlib import Path

BASE_DIR = Path.cwd()

graph_npz_file = (
    BASE_DIR
    / ".."
    / ".."
    / ".."
    / "data"
    / "processed"
    / "graph_structure"
    / "graph_structure_20260327_152434.npz"
).resolve()

graph_npz = np.load(graph_npz_file, allow_pickle=True)

coordinates = graph_npz["coordinates"]
edge_index = graph_npz["edge_index"]

lats = coordinates[:, 0]
lons = coordinates[:, 1]

print("Lat range:", lats.min(), lats.max())
print("Lon range:", lons.min(), lons.max())

# =========================
# 2. TẠO POLYGON (giống TomTom)
# =========================
polygon = Polygon([
    (lons.min(), lats.min()),
    (lons.max(), lats.min()),
    (lons.max(), lats.max()),
    (lons.min(), lats.max())
])

# =========================
# 3. LOAD OSM DATA
# =========================
print("Loading OSM from polygon...")

G = ox.graph_from_polygon(
    polygon,
    network_type="drive"
)

gdf_edges = ox.graph_to_gdfs(G, nodes=False)

# =========================
# 4. TẠO MAP
# =========================
center_lat = np.mean(lats)
center_lon = np.mean(lons)

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=14,
    tiles="cartodb positron"
)

# =========================
# 5. VẼ OSM ROAD (nền)
# =========================
for _, row in gdf_edges.iterrows():
    coords = [(lat, lon) for lon, lat in row.geometry.coords]

    folium.PolyLine(
        coords,
        color="#d2691e",
        weight=2,
        opacity=0.6
    ).add_to(m)

# =========================
# 6. VẼ GRAPH CỦA BẠN
# =========================

# edges
for i in range(edge_index.shape[1]):
    s = int(edge_index[0, i])
    e = int(edge_index[1, i])

    if 0 <= s < len(coordinates) and 0 <= e < len(coordinates):
        folium.PolyLine(
            [
                [coordinates[s, 0], coordinates[s, 1]],
                [coordinates[e, 0], coordinates[e, 1]],
            ],
            color="blue",  # nổi bật hơn OSM
            weight=2,
            opacity=0.9,
        ).add_to(m)

# nodes
for i in range(len(coordinates)):
    folium.CircleMarker(
        location=[coordinates[i, 0], coordinates[i, 1]],
        radius=3,
        color="red",
        fill=True
    ).add_to(m)

# =========================
# 7. FIT BOUNDS (QUAN TRỌNG)
# =========================
m.fit_bounds([
    [lats.min(), lons.min()],
    [lats.max(), lons.max()]
])

# =========================
# 8. SAVE
# =========================
m.save("final_map_osm_tomtom_aligned.html")

print("✅ DONE: final_map_osm_tomtom_aligned.html")


# =========================
# 5. MAP-MATCHING: KHỚP TOMTOM VÀO OSM
# =========================
print("Đang tìm nút OSM gần nhất cho các điểm TomTom...")
# Tìm danh sách ID của các nút OSM gần nhất với tọa độ (Lon, Lat) của TomTom
# Lưu ý: ox.distance.nearest_nodes nhận X=Lon, Y=Lat
osm_nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)

print("Đang tìm đường đi thực tế (Shortest Path) trên mạng lưới OSM...")
matched_paths = []

for i in range(edge_index.shape[1]):
    s = int(edge_index[0, i])
    e = int(edge_index[1, i])

    if 0 <= s < len(coordinates) and 0 <= e < len(coordinates):
        osm_u = osm_nodes[s]
        osm_v = osm_nodes[e]
        
        # Nếu điểm đầu và cuối không trùng vào cùng 1 nút OSM
        if osm_u != osm_v:
            try:
                # Tìm đường đi ngắn nhất trên đồ thị OSM (bám theo đường bộ)
                route = nx.shortest_path(G, osm_u, osm_v, weight='length')
                matched_paths.append(route)
            except nx.NetworkXNoPath:
                # Bỏ qua nếu OSM không có đường nối (ví dụ: đường 1 chiều cấm đi ngược)
                pass

# =========================
# 6. VẼ CÁC ĐOẠN ĐƯỜNG ĐÃ ĐƯỢC KHỚP
# =========================
print(f"Đã khớp thành công {len(matched_paths)} đoạn đường. Đang vẽ bản đồ...")

for route in matched_paths:
    route_coords = []
    # Duyệt qua từng cặp node liên tiếp trên route để lấy hình dáng đường cong
    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.get_edge_data(u, v)[0]
        
        if 'geometry' in edge_data:
            # Nếu đoạn đường cong (có chứa geometry từ OSM)
            xs, ys = edge_data['geometry'].xy
            route_coords.extend(list(zip(ys, xs))) # Folium yêu cầu [Lat, Lon]
        else:
            # Nếu đoạn đường là đường thẳng giữa 2 nút
            route_coords.append([G.nodes[u]['y'], G.nodes[u]['x']])
            route_coords.append([G.nodes[v]['y'], G.nodes[v]['x']])

    # Vẽ đường thực tế bám sát OSM
    folium.PolyLine(
        route_coords,
        color="purple", # Màu Cyan sáng để dễ nhìn trên nền tối
        weight=4,
        opacity=0.8
    ).add_to(m)

# (Tùy chọn) Vẽ các điểm TomTom gốc màu đỏ để bạn thấy nó đã được "hút" vào đường như thế nào
for i in range(len(coordinates)):
    folium.CircleMarker(
        location=[coordinates[i, 0], coordinates[i, 1]],
        radius=2,
        color="#ff1744", # Đỏ
        fill=True
    ).add_to(m)

# =========================
# 7. FIT BOUNDS & SAVE
# =========================
m.fit_bounds([
    [lats.min(), lons.min()],
    [lats.max(), lons.max()]
])

m.save("only_matched_osm.html")
print("✅ DONE: only_matched_osm.html")