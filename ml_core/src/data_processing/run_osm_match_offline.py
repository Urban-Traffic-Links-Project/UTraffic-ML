# ml_core/src/data_processing/run_osm_match_offline.py
"""
Offline TomTom -> OSM match checker.

Mục tiêu:
- Bỏ Kafka.
- Đọc trực tiếp dataset/tomtom_newest.zip.
- Lọc dữ liệu 06:00-12:00.
- Build OSM skeleton theo polygon hardcode Quận 1.
- Match TomTom segment geometry vào OSM edges bằng logic shortest path sẵn có.
- Xuất file để kiểm tra sau match còn bao nhiêu đoạn đường OSM.
- Lưu mapping để backend / pipeline sau dùng lại.

Vị trí đặt file khuyến nghị:
    UTraffic-ML/ml_core/src/data_processing/run_osm_match_offline.py

Chạy từ thư mục gốc project:
    cd C:/AI/Thesis/UTraffic-ML
    python ml_core/src/data_processing/run_osm_match_offline.py

Input mặc định:
    UTraffic-ML/dataset/tomtom_newest.zip

Output mặc định:
    UTraffic-ML/ml_core/src/data_processing/outputs/branchA/
"""

from __future__ import annotations

import json
import math
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATH CONFIG
# =============================================================================

THIS_FILE = Path(__file__).resolve()

# File này nằm tại: UTraffic-ML/ml_core/src/data_processing/run_osm_match_offline.py
DATA_PROCESSING_DIR = THIS_FILE.parent                 # .../ml_core/src/data_processing
SRC_ROOT = DATA_PROCESSING_DIR.parent                  # .../ml_core/src
ML_CORE_ROOT = SRC_ROOT.parent                         # .../ml_core
PROJECT_ROOT = ML_CORE_ROOT.parent                     # .../UTraffic-ML

# Để import được data_processing.graph.*
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_processing.graph.osm_graph_builder import OSMGraphBuilder
from data_processing.graph.map_matcher import (
    TomTomOSMMapMatcher,
    TOMTOM_FEATURE_COLS,
    EDGE_FEATURE_COLS,
)


# =============================================================================
# USER CONFIG
# =============================================================================

INPUT_ZIP = PROJECT_ROOT / "dataset" / "tomtom_newest.zip"

# Theo yêu cầu: output trong src/data_processing/outputs/branchA
OUTPUT_BASE = DATA_PROCESSING_DIR / "outputs" / "branchA"

# Giữ đúng cấu hình tomtom_6to12_w10
START_HOUR = 6
END_HOUR = 12       # lấy slot có giờ bắt đầu < 12
WINDOW_NAME = "w10"

# Ngưỡng match. Có thể thử 30, 50, 80, 100 để so sánh coverage.
MATCH_THRESHOLD_M = 50.0

# Nếu time_set không parse được giờ, mặc định vẫn giữ lại để tránh mất dữ liệu.
# Sau khi chạy, xem matching_summary.json để biết số record unknown time.
KEEP_UNKNOWN_TIME_SLOT = True

# Chỉ dùng cho bước kiểm tra số đoạn sau match.
# Chưa build full temporal tensor để tránh rất chậm.
BUILD_TEMPORAL_TENSOR = False


# =============================================================================
# GEOMETRY — HARDCODE QUẬN 1 NHƯ PIPELINE CŨ
# =============================================================================

def get_geometry_quan1() -> Dict[str, Any]:
    """
    GeoJSON Polygon cho Quận 1, HCMC.
    Thứ tự điểm: [longitude, latitude].
    """
    return {
        "type": "Polygon",
        "coordinates": [[
            [106.6750, 10.7600],
            [106.7150, 10.7600],
            [106.7150, 10.8050],
            [106.6750, 10.8050],
            [106.6750, 10.7600],
        ]]
    }


# =============================================================================
# BASIC UTILS
# =============================================================================

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float(x: Any, default: float = np.nan) -> float:
    if x is None:
        return default
    try:
        if isinstance(x, str):
            x = x.strip()
            if not x:
                return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = -1) -> int:
    if x is None:
        return default
    try:
        return int(float(x))
    except Exception:
        return default


def get_first(d: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def json_dumps_safe(obj: Any) -> str:
    class NpEncoder(json.JSONEncoder):
        def default(self, o: Any):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    return json.dumps(obj, ensure_ascii=False, indent=2, cls=NpEncoder)


def save_npz_dict(
    data: Dict[str, Any],
    output_dir: Path,
    dataset_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save dictionary arrays theo style:
        output_dir/dataset_name_YYYYmmdd_HHMMSS.npz
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset_name}_{now_stamp()}.npz"

    save_dict: Dict[str, np.ndarray] = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = v
        else:
            save_dict[k] = np.array(v)

    if metadata is not None:
        save_dict["_metadata"] = np.array([json_dumps_safe(metadata)])

    np.savez_compressed(out_path, **save_dict)
    return out_path


def save_dataframe_npz(
    df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save dataframe thành NPZ, mỗi cột là một array.
    """
    data = {col: df[col].to_numpy() for col in df.columns}
    meta = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
    }
    if metadata:
        meta.update(metadata)
    return save_npz_dict(data, output_dir, dataset_name, meta)


# =============================================================================
# TIME FILTER
# =============================================================================

_TIME_HHMM_COLON = re.compile(r"(?<!\d)([0-2]?\d):([0-5]\d)(?!\d)")
_TIME_HHMM_COMPACT = re.compile(r"(?<!\d)([0-2]\d)([0-5]\d)(?!\d)")


def parse_start_minutes(time_label: Any) -> Optional[int]:
    """
    Parse giờ bắt đầu từ time_set name.

    Hỗ trợ:
        "06:00-06:15"
        "06:00"
        "Slot_0600"
        "0600"
    """
    if time_label is None:
        return None

    s = str(time_label)

    m = _TIME_HHMM_COLON.search(s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23:
            return hh * 60 + mm

    m = _TIME_HHMM_COMPACT.search(s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23:
            return hh * 60 + mm

    return None


def keep_time_slot(time_label: Any) -> Tuple[bool, bool]:
    """
    Returns:
        keep, is_unknown
    """
    mins = parse_start_minutes(time_label)
    if mins is None:
        return KEEP_UNKNOWN_TIME_SLOT, True

    start = START_HOUR * 60
    end = END_HOUR * 60
    return start <= mins < end, False


# =============================================================================
# ZIP READER + TOMTOM JSON PARSER
# =============================================================================

def list_job_json_files(zip_path: Path) -> List[str]:
    if not zip_path.exists():
        raise FileNotFoundError(f"Không tìm thấy input zip: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [
            name for name in zf.namelist()
            if name.lower().endswith(".json")
            and Path(name).name.startswith("job_")
            and Path(name).name.endswith("_results.json")
        ]

    return sorted(names)


def load_json_from_zip(zip_path: Path, member_name: str) -> Dict[str, Any]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name, "r") as f:
            raw = f.read()
    return json.loads(raw.decode("utf-8"))


def build_id_name_map(items: List[Dict[str, Any]], value_keys: List[str]) -> Dict[str, Any]:
    """
    Convert list object TomTom timeSets/dateRanges thành map id -> name/from.
    """
    result: Dict[str, Any] = {}
    for item in items or []:
        item_id = get_first(item, ["@id", "id", "key", "name"])
        if item_id is None:
            continue

        value = get_first(item, value_keys)
        if value is None:
            value = item_id

        result[str(item_id)] = value

    return result


def extract_shape_points(segment: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """
    Returns:
        lats, lons
    """
    shape = segment.get("shape") or segment.get("points") or segment.get("coordinates") or []
    lats: List[float] = []
    lons: List[float] = []

    for p in shape:
        lat = None
        lon = None

        if isinstance(p, dict):
            lat = get_first(p, ["latitude", "lat", "y"])
            lon = get_first(p, ["longitude", "lon", "lng", "x"])
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            # GeoJSON thường [lon, lat]
            lon = p[0]
            lat = p[1]

        lat_f = safe_float(lat)
        lon_f = safe_float(lon)

        if not math.isnan(lat_f) and not math.isnan(lon_f):
            lats.append(lat_f)
            lons.append(lon_f)

    return lats, lons


def extract_records_from_job_json(
    data: Dict[str, Any],
    source_file: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Parse 1 file job_*_results.json thành records dạng traffic_features thô.

    Mỗi record = 1 segment_id tại 1 date_range + 1 time_set.
    """
    time_sets_map = build_id_name_map(
        data.get("timeSets", []),
        value_keys=["name", "from", "time", "label"],
    )
    date_ranges_map = build_id_name_map(
        data.get("dateRanges", []),
        value_keys=["from", "date", "name", "startDate"],
    )

    network = data.get("network", {})
    segments = network.get("segmentResults", []) or []

    records: List[Dict[str, Any]] = []
    stats = {
        "segments_seen": 0,
        "segments_without_shape": 0,
        "records_seen": 0,
        "records_kept": 0,
        "records_skipped_time": 0,
        "records_unknown_time": 0,
    }

    job_id_match = re.search(r"job_(.*?)_results\.json", Path(source_file).name)
    job_id = job_id_match.group(1) if job_id_match else None

    for segment in segments:
        stats["segments_seen"] += 1

        lats, lons = extract_shape_points(segment)
        if not lats or not lons:
            stats["segments_without_shape"] += 1
            continue

        mean_lat = float(np.mean(lats))
        mean_lon = float(np.mean(lons))
        start_lat = float(lats[0])
        start_lon = float(lons[0])
        end_lat = float(lats[-1])
        end_lon = float(lons[-1])

        speed_limit = safe_float(
            get_first(segment, ["speedLimit", "speed_limit", "freeFlowSpeed"]),
            default=np.nan,
        )

        time_results = segment.get("segmentTimeResults", []) or []

        for tr in time_results:
            stats["records_seen"] += 1

            time_id = get_first(tr, ["timeSet", "time_set", "timeSetId"])
            date_id = get_first(tr, ["dateRange", "date_range", "dateRangeId"])

            time_set_name = time_sets_map.get(str(time_id), time_id)
            date_from = date_ranges_map.get(str(date_id), date_id)

            keep, unknown_time = keep_time_slot(time_set_name)
            if unknown_time:
                stats["records_unknown_time"] += 1

            if not keep:
                stats["records_skipped_time"] += 1
                continue

            average_speed = safe_float(get_first(tr, ["averageSpeed", "average_speed"]))
            harmonic_average_speed = safe_float(
                get_first(tr, ["harmonicAverageSpeed", "harmonic_average_speed"])
            )
            median_speed = safe_float(get_first(tr, ["medianSpeed", "median_speed"]))
            std_speed = safe_float(
                get_first(tr, ["standardDeviationSpeed", "std_speed", "speedStandardDeviation"])
            )

            average_travel_time = safe_float(
                get_first(tr, ["averageTravelTime", "average_travel_time"])
            )
            median_travel_time = safe_float(
                get_first(tr, ["medianTravelTime", "median_travel_time"])
            )
            travel_time_std = safe_float(
                get_first(tr, ["travelTimeStandardDeviation", "travel_time_std"])
            )
            travel_time_ratio = safe_float(
                get_first(tr, ["travelTimeRatio", "travel_time_ratio"])
            )
            sample_size = safe_float(get_first(tr, ["sampleSize", "sample_size"]))

            if not math.isnan(speed_limit) and speed_limit > 0 and not math.isnan(average_speed):
                speed_limit_ratio = average_speed / speed_limit
                congestion_index = max(0.0, 1.0 - speed_limit_ratio)
            else:
                speed_limit_ratio = np.nan
                congestion_index = np.nan

            record = {
                "source_file": source_file,
                "job_id": job_id,
                "segment_id": get_first(segment, ["segmentId", "segment_id"]),
                "new_segment_id": get_first(segment, ["newSegmentId", "new_segment_id"]),
                "street_name": get_first(segment, ["streetName", "street_name", "name"]),
                "distance": safe_float(get_first(segment, ["distance", "length"])),
                "frc": get_first(segment, ["frc", "functionalRoadClass"]),
                "speed_limit": speed_limit,

                "date_from": date_from,
                "date_range": date_from,
                "time_set": time_set_name,

                "harmonic_average_speed": harmonic_average_speed,
                "median_speed": median_speed,
                "average_speed": average_speed,
                "std_speed": std_speed,
                "average_travel_time": average_travel_time,
                "median_travel_time": median_travel_time,
                "travel_time_std": travel_time_std,
                "travel_time_ratio": travel_time_ratio,
                "sample_size": sample_size,
                "speed_limit_ratio": speed_limit_ratio,
                "congestion_index": congestion_index,

                # raw coordinates, bắt buộc cho OSM matching
                "raw_latitude": mean_lat,
                "raw_longitude": mean_lon,
                "raw_lat_start": start_lat,
                "raw_lon_start": start_lon,
                "raw_lat_end": end_lat,
                "raw_lon_end": end_lon,
            }

            records.append(record)
            stats["records_kept"] += 1

    return records, stats


def load_tomtom_zip_to_dataframe(zip_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    job_files = list_job_json_files(zip_path)
    if not job_files:
        raise RuntimeError(f"Không tìm thấy job_*_results.json trong {zip_path}")

    all_records: List[Dict[str, Any]] = []
    total_stats: Dict[str, Any] = {
        "zip_path": str(zip_path),
        "num_job_files": len(job_files),
        "job_files_preview": job_files[:10],
        "segments_seen": 0,
        "segments_without_shape": 0,
        "records_seen": 0,
        "records_kept": 0,
        "records_skipped_time": 0,
        "records_unknown_time": 0,
    }

    print(f"Found {len(job_files)} job JSON files in zip.")

    for i, member in enumerate(job_files, start=1):
        data = load_json_from_zip(zip_path, member)
        records, stats = extract_records_from_job_json(data, member)
        all_records.extend(records)

        for k in [
            "segments_seen",
            "segments_without_shape",
            "records_seen",
            "records_kept",
            "records_skipped_time",
            "records_unknown_time",
        ]:
            total_stats[k] += stats[k]

        if i % 5 == 0 or i == len(job_files):
            print(
                f"  Parsed {i}/{len(job_files)} files | "
                f"kept records={len(all_records):,}"
            )

    df = pd.DataFrame(all_records)
    if df.empty:
        raise RuntimeError("Không có record nào sau khi đọc zip và lọc time.")

    # Chuẩn hóa kiểu dữ liệu cơ bản
    df["segment_id"] = df["segment_id"].astype(str)
    df["time_set"] = df["time_set"].astype(str)
    df["date_from"] = df["date_from"].astype(str)
    df["date_range"] = df["date_range"].astype(str)

    numeric_cols = [
        "distance", "speed_limit",
        "harmonic_average_speed", "median_speed", "average_speed", "std_speed",
        "average_travel_time", "median_travel_time", "travel_time_std",
        "travel_time_ratio", "sample_size",
        "speed_limit_ratio", "congestion_index",
        "raw_latitude", "raw_longitude",
        "raw_lat_start", "raw_lon_start", "raw_lat_end", "raw_lon_end",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    total_stats["final_records"] = int(len(df))
    total_stats["unique_tomtom_segments"] = int(df["segment_id"].nunique())
    total_stats["unique_dates"] = int(df["date_from"].nunique())
    total_stats["unique_time_slots"] = int(df["time_set"].nunique())
    total_stats["time_slots_preview"] = sorted(df["time_set"].dropna().unique().tolist())[:20]

    return df, total_stats


# =============================================================================
# SEGMENT-LEVEL MATCH INPUT
# =============================================================================

def build_segment_level_df(traffic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo dataframe 1 dòng / TomTom segment để match geometry đúng 1 lần.

    Dữ liệu động lấy mean theo thời gian, chỉ để tạo edge_features preview.
    Phần quan trọng nhất là raw_lat_start/raw_lon_start/raw_lat_end/raw_lon_end.
    """
    df = traffic_df.copy()

    agg: Dict[str, Any] = {
        "new_segment_id": "first",
        "street_name": "first",
        "frc": "first",
        "distance": "first",
        "speed_limit": "first",
        "raw_latitude": "first",
        "raw_longitude": "first",
        "raw_lat_start": "first",
        "raw_lon_start": "first",
        "raw_lat_end": "first",
        "raw_lon_end": "first",
    }

    for col in TOMTOM_FEATURE_COLS:
        if col in df.columns:
            agg[col] = "mean"

    segment_df = (
        df
        .groupby("segment_id", as_index=False, dropna=False)
        .agg(agg)
    )

    # Bỏ các cột thời gian để map_matcher group đúng theo segment_id
    for col in ["date_from", "date_range", "date", "time_set"]:
        if col in segment_df.columns:
            segment_df = segment_df.drop(columns=[col])

    return segment_df


# =============================================================================
# MATCH METADATA
# =============================================================================

def add_static_osm_attrs(
    edge_meta: pd.DataFrame,
    matcher: TomTomOSMMapMatcher,
) -> pd.DataFrame:
    global_src = matcher.edge_index[0]
    global_dst = matcher.edge_index[1]

    lengths = []
    maxspeeds = []
    lanes = []
    highway_types = []

    for u, v in zip(edge_meta["osm_u_idx"].values, edge_meta["osm_v_idx"].values):
        mask = (global_src == int(u)) & (global_dst == int(v))
        if mask.any():
            pos = int(np.where(mask)[0][0])
            lengths.append(float(matcher.edge_lengths[pos]))
            maxspeeds.append(float(matcher.edge_maxspeed[pos]))
            lanes.append(float(matcher.edge_lanes[pos]))
            highway_types.append(int(matcher.edge_highway_type[pos]))
        else:
            lengths.append(np.nan)
            maxspeeds.append(np.nan)
            lanes.append(np.nan)
            highway_types.append(-1)

    edge_meta["osm_length_m"] = lengths
    edge_meta["osm_maxspeed"] = maxspeeds
    edge_meta["osm_lanes"] = lanes
    edge_meta["osm_highway_type"] = highway_types
    return edge_meta


def compact_unique(values: pd.Series, max_items: int = 30) -> str:
    vals = []
    for v in values.dropna().astype(str).tolist():
        if v not in vals:
            vals.append(v)
        if len(vals) >= max_items:
            break
    return "|".join(vals)


def build_edge_metadata(
    matched_df: pd.DataFrame,
    matcher: TomTomOSMMapMatcher,
) -> pd.DataFrame:
    """
    1 dòng = 1 matched OSM directed edge.
    Đây là danh sách "node" cho Branch A/B nếu chọn node = matched OSM edge.
    """
    if matched_df.empty:
        return pd.DataFrame()

    edge_meta = (
        matched_df
        .groupby(["osm_u_idx", "osm_v_idx"], sort=False)
        .agg(
            tomtom_match_rows=("segment_id", "count"),
            tomtom_unique_segments=("segment_id", "nunique"),
            mean_match_dist_m=("match_dist_m", "mean"),
            median_match_dist_m=("match_dist_m", "median"),
            street_names=("street_name", compact_unique),
            tomtom_segment_ids=("segment_id", compact_unique),
        )
        .reset_index()
    )

    edge_meta["model_node_id"] = np.arange(len(edge_meta), dtype=np.int64)

    osm_u_ids = matcher.osm_node_ids[edge_meta["osm_u_idx"].astype(int).values]
    osm_v_ids = matcher.osm_node_ids[edge_meta["osm_v_idx"].astype(int).values]

    edge_meta["osm_u_id"] = osm_u_ids.astype(np.int64)
    edge_meta["osm_v_id"] = osm_v_ids.astype(np.int64)
    edge_meta["osm_edge_id"] = [
        f"{int(u)}_{int(v)}" for u, v in zip(edge_meta["osm_u_id"], edge_meta["osm_v_id"])
    ]

    u_coords = matcher.coordinates[edge_meta["osm_u_idx"].astype(int).values]
    v_coords = matcher.coordinates[edge_meta["osm_v_idx"].astype(int).values]

    edge_meta["u_lat"] = u_coords[:, 0]
    edge_meta["u_lon"] = u_coords[:, 1]
    edge_meta["v_lat"] = v_coords[:, 0]
    edge_meta["v_lon"] = v_coords[:, 1]
    edge_meta["mid_lat"] = (edge_meta["u_lat"] + edge_meta["v_lat"]) / 2.0
    edge_meta["mid_lon"] = (edge_meta["u_lon"] + edge_meta["v_lon"]) / 2.0

    edge_meta = add_static_osm_attrs(edge_meta, matcher)

    # Sắp xếp cột cho dễ đọc
    first_cols = [
        "model_node_id",
        "osm_edge_id",
        "osm_u_id", "osm_v_id",
        "osm_u_idx", "osm_v_idx",
        "u_lat", "u_lon", "v_lat", "v_lon", "mid_lat", "mid_lon",
        "osm_length_m", "osm_maxspeed", "osm_lanes", "osm_highway_type",
        "tomtom_unique_segments", "tomtom_match_rows",
        "mean_match_dist_m", "median_match_dist_m",
        "street_names", "tomtom_segment_ids",
    ]
    other_cols = [c for c in edge_meta.columns if c not in first_cols]
    return edge_meta[first_cols + other_cols]


def add_model_node_mapping_to_graph_result(
    graph_result: Dict[str, np.ndarray],
    edge_meta: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Bổ sung mapping node = matched OSM edge để downstream dùng.
    """
    if edge_meta.empty:
        return graph_result

    graph_result = dict(graph_result)
    graph_result["model_node_ids"] = edge_meta["model_node_id"].to_numpy(dtype=np.int64)
    graph_result["model_node_osm_edge_id"] = edge_meta["osm_edge_id"].astype(str).to_numpy()
    graph_result["model_node_osm_u_id"] = edge_meta["osm_u_id"].to_numpy(dtype=np.int64)
    graph_result["model_node_osm_v_id"] = edge_meta["osm_v_id"].to_numpy(dtype=np.int64)
    graph_result["model_node_osm_u_idx"] = edge_meta["osm_u_idx"].to_numpy(dtype=np.int64)
    graph_result["model_node_osm_v_idx"] = edge_meta["osm_v_idx"].to_numpy(dtype=np.int64)
    graph_result["model_node_mid_lat"] = edge_meta["mid_lat"].to_numpy(dtype=np.float64)
    graph_result["model_node_mid_lon"] = edge_meta["mid_lon"].to_numpy(dtype=np.float64)
    graph_result["model_node_osm_length_m"] = edge_meta["osm_length_m"].to_numpy(dtype=np.float32)
    graph_result["model_node_tomtom_unique_segments"] = edge_meta["tomtom_unique_segments"].to_numpy(dtype=np.int64)

    return graph_result


def count_undirected_edges(edge_meta: pd.DataFrame) -> int:
    if edge_meta.empty:
        return 0
    pairs = set()
    for u, v in zip(edge_meta["osm_u_id"].values, edge_meta["osm_v_id"].values):
        a, b = sorted([int(u), int(v)])
        pairs.add((a, b))
    return len(pairs)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("OFFLINE TOMTOM -> OSM MATCH CHECKER")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input zip    : {INPUT_ZIP}")
    print(f"Output base  : {OUTPUT_BASE}")
    print(f"Time filter  : {START_HOUR:02d}:00 <= time < {END_HOUR:02d}:00")
    print(f"Threshold    : {MATCH_THRESHOLD_M} m")
    print("=" * 80)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load TomTom JSON from zip
    # -------------------------------------------------------------------------
    traffic_df, load_stats = load_tomtom_zip_to_dataframe(INPUT_ZIP)

    traffic_features_dir = OUTPUT_BASE / "traffic_features"
    traffic_npz_path = save_dataframe_npz(
        traffic_df,
        output_dir=traffic_features_dir,
        dataset_name="traffic_features",
        metadata={
            "purpose": "offline extracted TomTom traffic features from zip",
            "time_filter": f"{START_HOUR}:00-{END_HOUR}:00",
            "window_name": WINDOW_NAME,
        },
    )

    print("\n[1] TomTom features")
    print(f"    Records kept           : {len(traffic_df):,}")
    print(f"    Unique TomTom segments : {traffic_df['segment_id'].nunique():,}")
    print(f"    Unique dates           : {traffic_df['date_from'].nunique():,}")
    print(f"    Unique time slots      : {traffic_df['time_set'].nunique():,}")
    print(f"    Saved traffic features : {traffic_npz_path}")

    # -------------------------------------------------------------------------
    # 2. Build segment-level dataframe for geometry matching
    # -------------------------------------------------------------------------
    segment_df = build_segment_level_df(traffic_df)
    segment_csv_dir = OUTPUT_BASE / "match_summary"
    segment_csv_dir.mkdir(parents=True, exist_ok=True)
    segment_csv_path = segment_csv_dir / "tomtom_segment_level_for_match.csv"
    segment_df.to_csv(segment_csv_path, index=False, encoding="utf-8-sig")

    print("\n[2] Segment-level match input")
    print(f"    Segment rows           : {len(segment_df):,}")
    print(f"    Saved segment CSV      : {segment_csv_path}")

    # -------------------------------------------------------------------------
    # 3. Build OSM skeleton
    # -------------------------------------------------------------------------
    print("\n[3] Build OSM graph")
    geometry = get_geometry_quan1()

    osm_builder = OSMGraphBuilder(
        output_dir=OUTPUT_BASE,
        cache_graph=True,
    )
    osm_graph_data = osm_builder.build_from_polygon(
        polygon=geometry,
        simplify=True,
        retain_all=False,
        output_name="osm_graph",
    )

    G_nx = osm_builder.G
    if G_nx is None:
        raise RuntimeError("OSMGraphBuilder.G is None. Không thể match.")

    print(f"    OSM nodes              : {len(osm_graph_data['osm_node_ids']):,}")
    print(f"    OSM directed edges     : {osm_graph_data['edge_index'].shape[1]:,}")
    print(f"    OSM cache dir          : {OUTPUT_BASE / 'osm_cache'}")
    print(f"    OSM graph dir          : {OUTPUT_BASE / 'osm_graph'}")

    # -------------------------------------------------------------------------
    # 4. Match TomTom segments -> OSM edges
    # -------------------------------------------------------------------------
    print("\n[4] Match TomTom segments -> OSM edges")

    matcher = TomTomOSMMapMatcher(
        osm_graph_data=osm_graph_data,
        match_threshold_m=MATCH_THRESHOLD_M,
        aggregate_method="mean",
    )

    prepared_df = matcher._prepare_traffic_df(segment_df, time_slot=None)
    matched_df = matcher._map_match_segments(prepared_df, G_nx)

    if matched_df.empty:
        raise RuntimeError(
            "Không match được segment nào. Hãy thử tăng MATCH_THRESHOLD_M lên 80 hoặc 100."
        )

    matched_csv_path = segment_csv_dir / "tomtom_to_osm_edge_matches.csv.gz"
    matched_df.to_csv(matched_csv_path, index=False, encoding="utf-8-sig", compression="gzip")

    edge_meta = build_edge_metadata(matched_df, matcher)
    edge_meta_csv_path = segment_csv_dir / "matched_osm_edge_metadata.csv"
    edge_meta.to_csv(edge_meta_csv_path, index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # 5. Build static graph_structure preview
    # -------------------------------------------------------------------------
    print("\n[5] Build graph_structure preview")
    graph_result = matcher._build_matched_subgraph(matched_df)
    graph_result = add_model_node_mapping_to_graph_result(graph_result, edge_meta)

    graph_npz_path = save_npz_dict(
        graph_result,
        output_dir=OUTPUT_BASE / "graph_structure",
        dataset_name="graph_structure",
        metadata={
            "purpose": "static matched graph preview for counting matched OSM road segments",
            "note": "For Branch A/B, model_node_id corresponds to one matched OSM directed edge.",
            "match_threshold_m": MATCH_THRESHOLD_M,
            "time_filter": f"{START_HOUR}:00-{END_HOUR}:00",
            "model_node_definition": "matched_osm_directed_edge",
            "num_model_nodes": int(len(edge_meta)),
            "num_matched_osm_directed_edges": int(len(edge_meta)),
            "num_matched_osm_undirected_edges": int(count_undirected_edges(edge_meta)),
        },
    )

    # -------------------------------------------------------------------------
    # 6. Summary
    # -------------------------------------------------------------------------
    total_tomtom_segments = int(segment_df["segment_id"].nunique())
    matched_tomtom_segments = int(matched_df["segment_id"].nunique())
    matched_osm_directed_edges = int(len(edge_meta))
    matched_osm_undirected_edges = int(count_undirected_edges(edge_meta))

    summary = {
        **load_stats,
        "output_base": str(OUTPUT_BASE),
        "time_filter": {
            "start_hour": START_HOUR,
            "end_hour": END_HOUR,
            "window_name": WINDOW_NAME,
            "keep_unknown_time_slot": KEEP_UNKNOWN_TIME_SLOT,
        },
        "match": {
            "match_threshold_m": MATCH_THRESHOLD_M,
            "total_tomtom_segments": total_tomtom_segments,
            "matched_tomtom_segments": matched_tomtom_segments,
            "coverage_tomtom_segments": matched_tomtom_segments / max(total_tomtom_segments, 1),
            "matched_osm_directed_edges": matched_osm_directed_edges,
            "matched_osm_undirected_edges": matched_osm_undirected_edges,
            "model_node_definition": "matched_osm_directed_edge",
            "model_nodes_for_branchA_B": matched_osm_directed_edges,
        },
        "osm_graph": {
            "osm_nodes": int(len(osm_graph_data["osm_node_ids"])),
            "osm_directed_edges": int(osm_graph_data["edge_index"].shape[1]),
        },
        "files": {
            "traffic_features_npz": str(traffic_npz_path),
            "tomtom_segment_level_csv": str(segment_csv_path),
            "full_match_table_csv_gz": str(matched_csv_path),
            "matched_osm_edge_metadata_csv": str(edge_meta_csv_path),
            "graph_structure_npz": str(graph_npz_path),
            "osm_cache_dir": str(OUTPUT_BASE / "osm_cache"),
            "osm_graph_dir": str(OUTPUT_BASE / "osm_graph"),
            "graph_structure_dir": str(OUTPUT_BASE / "graph_structure"),
        },
    }

    summary_path = segment_csv_dir / "matching_summary.json"
    summary_path.write_text(json_dumps_safe(summary), encoding="utf-8")

    print("\n" + "=" * 80)
    print("MATCH SUMMARY")
    print("=" * 80)
    print(f"TomTom segments before match       : {total_tomtom_segments:,}")
    print(f"TomTom segments matched            : {matched_tomtom_segments:,}")
    print(f"Coverage                           : {summary['match']['coverage_tomtom_segments']:.2%}")
    print(f"Matched OSM directed edges         : {matched_osm_directed_edges:,}")
    print(f"Matched OSM undirected road pieces : {matched_osm_undirected_edges:,}")
    print(f"Model nodes for Branch A/B         : {matched_osm_directed_edges:,}")
    print("-" * 80)
    print(f"Summary JSON                       : {summary_path}")
    print(f"Edge metadata CSV                  : {edge_meta_csv_path}")
    print(f"Graph structure NPZ                : {graph_npz_path}")
    print("=" * 80)

    if load_stats.get("records_unknown_time", 0) > 0:
        print("\nWARNING:")
        print(
            f"  Có {load_stats['records_unknown_time']:,} records không parse được time_set. "
            "Vì KEEP_UNKNOWN_TIME_SLOT=True nên các record này vẫn được giữ lại."
        )
        print("  Hãy mở matching_summary.json để xem time_slots_preview.")
        print("  Nếu thấy dữ liệu bị lấy quá rộng, cần chỉnh hàm parse_start_minutes().")


if __name__ == "__main__":
    main()
