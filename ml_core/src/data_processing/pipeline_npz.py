from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .collectors.tomtom_collector import TomTomTrafficDataCollector
from .streaming.kafka_producer import TrafficDataProducer
from .streaming.kafka_consumer import RawDataConsumer, ValidatedDataConsumer
from .preprocessors.data_validator import DataValidator
from .preprocessors.data_cleaner import DataCleaner
from .preprocessors.feature_extractor import (
    FeatureExtractor,
    FINAL_FEATURE_COLS,
    NORMALIZE_COLS,
    MINMAX_COLS,
)
from .preprocessors.data_normalizer import DataNormalizer
from .graph.osm_graph_builder import OSMGraphBuilder
from .graph.map_matcher import TomTomOSMMapMatcher
from .storage.npz_storage import NPZWriter, NPZReader

from utils.config import config
from utils.logger import setup_logger


class TrafficDataPipelineNPZ:
    """
    Pipeline xử lý dữ liệu giao thông theo Kappa Architecture.
    OSM Skeleton + TomTom Features hybrid.
    """

    def __init__(self):
        self.logger = setup_logger(
            "TrafficDataPipelineNPZ", config.log_file, config.log_level
        )

        # Core components
        self.collector = TomTomTrafficDataCollector()
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.feature_extractor = FeatureExtractor()
        self.normalizer = DataNormalizer()

        # Storage
        self.npz_writer = NPZWriter()
        self.npz_reader = NPZReader()

        # Batch accumulator
        self.accumulated_data: List[Dict] = []
        self.batch_size = 10_000

        # OSM graph (populated after build_osm_skeleton)
        self._osm_builder: Optional[OSMGraphBuilder] = None
        self._osm_graph_data: Optional[Dict[str, np.ndarray]] = None

        self.logger.info("✅ Pipeline (OSM+TomTom hybrid) initialized")

    # =========================================================================
    # STAGE 1 — TomTom Collection
    # =========================================================================

    def run_31_days_collection(
        self,
        geometry: Dict,
        start_date: str,
    ) -> Optional[List[str]]:
        """Stage 1: Thu thập 31 ngày từ TomTom API."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: 31-DAY SEQUENTIAL DATA COLLECTION")
        self.logger.info("=" * 60)

        job_ids, _ = self.collector.collect_31_days_sequential(
            geometry=geometry,
            start_date_str=start_date,
        )

        if not job_ids:
            self.logger.error("No jobs created")
            return None

        self.logger.info(f"✅ Stage 1 complete: {len(job_ids)} jobs")
        return job_ids

    # =========================================================================
    # STAGE 2A — Kafka Streaming Ingestion
    # =========================================================================

    def run_streaming_ingestion_from_jobs(self, job_ids: List[str]):
        """Stage 2a: Đưa dữ liệu từ file JSON vào Kafka stream."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2a: STREAMING INGESTION")
        self.logger.info("=" * 60)

        total_sent = 0
        for idx, job_id in enumerate(job_ids):
            n = self._ingest_one_job_to_stream(
                job_id, day_index=idx + 1, total_days=len(job_ids)
            )
            total_sent += n

        self.logger.info(
            f"✅ Stage 2a complete: {total_sent} segments from {len(job_ids)} days"
        )

    def _ingest_one_job_to_stream(
        self,
        job_id: str,
        day_index: Optional[int] = None,
        total_days: Optional[int] = None,
    ) -> int:
        """Đọc 1 file job_*_results.json và push vào Kafka."""
        result_file = (
            config.data.raw_dir / "tomtom_stats_frc5" / f"job_{job_id}_results.json"
        )

        if not result_file.exists():
            self.logger.error(f"Result file not found: {result_file}")
            return 0

        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        sent_count = 0
        with TrafficDataProducer() as producer:
            network = data.get("network", {})
            for segment in network.get("segmentResults", []):
                message = {
                    "job_id": job_id,
                    "segment": segment,
                    "job_name": data.get("jobName"),
                    "date_ranges": data.get("dateRanges"),
                    "time_sets": data.get("timeSets"),
                }
                producer.send_raw_data(message, key=str(segment.get("segmentId")))
                sent_count += 1
            producer.flush()

        label = f" (day {day_index}/{total_days})" if day_index and total_days else ""
        self.logger.info(f"  Sent {sent_count} segments for job {job_id}{label}")
        return sent_count

    # =========================================================================
    # STAGE 2B — OSM Skeleton (MỚI)
    # =========================================================================

    def build_osm_skeleton(
        self,
        geometry: Dict,
        output_name: str = "osm_graph",
        force_rebuild: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Stage 2b: Xây đồ thị OSM từ polygon geometry.

        Chạy SONG SONG với Stage 2a (independent, không cần Kafka).
        Kết quả lưu cache GraphML để không cần tải lại OSM mỗi lần chạy.

        Args:
            geometry     : GeoJSON Polygon (cùng format với TomTom collector).
            output_name  : Tên NPZ output.
            force_rebuild: Bỏ qua cache, tải lại từ OSM.

        Returns:
            Dict với node_ids, coordinates, edge_index, adjacency_matrix, ...
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2b: BUILD OSM SKELETON GRAPH")
        self.logger.info("=" * 60)

        # Thử load từ NPZ cache trước
        if not force_rebuild:
            cached = OSMGraphBuilder.load_latest(output_name, config.data.processed_dir)
            if cached is not None:
                self.logger.info(
                    f"✅ Loaded cached OSM graph: "
                    f"{len(cached['osm_node_ids'])} nodes, "
                    f"{cached['edge_index'].shape[1]} directed edges"
                )
                self._osm_graph_data = cached
                return cached

        self._osm_builder = OSMGraphBuilder(
            output_dir=config.data.processed_dir,
            cache_graph=True,
        )

        self._osm_graph_data = self._osm_builder.build_from_polygon(
            polygon=geometry,
            simplify=True,
            retain_all=False,
            output_name=output_name,
        )

        self.logger.info(
            f"✅ Stage 2b complete: "
            f"{len(self._osm_graph_data['osm_node_ids'])} OSM nodes, "
            f"{self._osm_graph_data['edge_index'].shape[1] // 2} undirected edges"
        )
        return self._osm_graph_data

    # =========================================================================
    # STAGE 3 — Validation
    # =========================================================================

    def run_validation_processing(self):
        """Stage 3: Validate segments từ Kafka, gửi sang traffic.validated."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: VALIDATION PROCESSING")
        self.logger.info("=" * 60)

        validated_count = 0

        def validate_processor(key, value, topic, partition, offset):
            nonlocal validated_count
            try:
                segment = value.get("segment", {})
                is_valid, errors = self.validator.validate_segment(segment)
                if not is_valid:
                    self.logger.debug(f"Invalid segment {key}: {errors}")
                    return True

                time_results = segment.get("segmentTimeResults", [])
                valid_trs = [
                    tr for tr in time_results
                    if self.validator.validate_time_result(tr)[0]
                ]

                if not valid_trs:
                    return True

                value["segment"]["segmentTimeResults"] = valid_trs

                with TrafficDataProducer() as producer:
                    producer.send_validated_data(value, key=key)

                validated_count += 1
                return True

            except Exception as e:
                self.logger.error(f"validate_processor error: {e}")
                return False

        with RawDataConsumer() as consumer:
            consumer.consume(validate_processor, max_messages=None)

        self.logger.info(f"✅ Stage 3 complete: {validated_count} segments validated")

    # =========================================================================
    # STAGE 4 — Feature Extraction
    # =========================================================================

    def run_feature_extraction(self):
        """Stage 4: Extract features (~28) từ validated data."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 4: FEATURE EXTRACTION (v2 — ~28 features)")
        self.logger.info("=" * 60)

        self.accumulated_data = []
        processed_count = 0

        def feature_processor(key, value, topic, partition, offset):
            nonlocal processed_count
            try:
                segment = value.get("segment", {})
                time_results = segment.get("segmentTimeResults", [])

                time_sets_map = {
                    ts.get("@id"): ts.get("name")
                    for ts in value.get("time_sets", [])
                }
                date_ranges_map = {
                    dr.get("@id"): dr.get("from")
                    for dr in value.get("date_ranges", [])
                }

                shape = segment.get("shape", [])
                lats = [p.get("latitude", 0) for p in shape]
                lons = [p.get("longitude", 0) for p in shape]
                mean_lat = float(np.mean(lats)) if lats else None
                mean_lon = float(np.mean(lons)) if lons else None
                # Start/end của shape để dùng trong shortest-path map matching
                start_lat = float(lats[0])  if lats else None
                start_lon = float(lons[0])  if lons else None
                end_lat   = float(lats[-1]) if lats else None
                end_lon   = float(lons[-1]) if lons else None

                for tr in time_results:
                    record = {
                        "segment_id":              segment.get("segmentId"),
                        "new_segment_id":           segment.get("newSegmentId"),
                        "street_name":              segment.get("streetName"),
                        "distance":                 segment.get("distance"),
                        "frc":                      segment.get("frc"),
                        "speed_limit":              segment.get("speedLimit"),
                        "time_set":                 time_sets_map.get(tr.get("timeSet")),
                        "date_from":                date_ranges_map.get(tr.get("dateRange")),
                        "harmonic_average_speed":   tr.get("harmonicAverageSpeed"),
                        "median_speed":             tr.get("medianSpeed"),
                        "average_speed":            tr.get("averageSpeed"),
                        "std_speed":                tr.get("standardDeviationSpeed"),
                        "average_travel_time":      tr.get("averageTravelTime"),
                        "median_travel_time":       tr.get("medianTravelTime"),
                        "travel_time_std":          tr.get("travelTimeStandardDeviation"),
                        "travel_time_ratio":        tr.get("travelTimeRatio"),
                        "sample_size":              tr.get("sampleSize"),
                        # Raw coordinates — KHÔNG normalize ở đây
                        "raw_latitude":             mean_lat,
                        "raw_longitude":            mean_lon,
                        # Tọa độ đầu/cuối segment để dùng shortest-path map matching
                        "raw_lat_start":            start_lat,
                        "raw_lon_start":            start_lon,
                        "raw_lat_end":              end_lat,
                        "raw_lon_end":              end_lon,
                    }
                    self.accumulated_data.append(record)

                if len(self.accumulated_data) >= self.batch_size:
                    self._process_and_save_batch()
                    processed_count += len(self.accumulated_data)
                    self.accumulated_data.clear()

                return True

            except Exception as e:
                self.logger.error(f"feature_processor error: {e}")
                return False

        with ValidatedDataConsumer() as consumer:
            consumer.consume(feature_processor, max_messages=None)

        if self.accumulated_data:
            self._process_and_save_batch()
            processed_count += len(self.accumulated_data)

        self.logger.info(
            f"✅ Stage 4 complete: {processed_count} records processed"
        )

    def _process_and_save_batch(self):
        """
        Clean + Extract features + Save batch vào NPZ.

        Normalization được DEFER sang export_for_model_training()
        để StandardScaler được fit một lần trên toàn bộ train set.
        """
        if not self.accumulated_data:
            return

        df = pd.DataFrame(self.accumulated_data)

        # 1. Clean
        df = self.cleaner.clean(df)
        if df.empty:
            self.logger.warning("Batch empty after cleaning, skipping.")
            return

        # 2. Extract features (v2 — ~28 features)
        df = self.feature_extractor.extract_all_features(df)

        # 3. Save (raw — chưa normalize)
        self.npz_writer.write_features(df)

        self.logger.info(
            f"  Saved batch: {len(df)} records, {len(df.columns)} columns"
        )

    # =========================================================================
    # STAGE 5 — Export for Model Training
    # =========================================================================

    def export_for_model_training(
        self,
        output_name: Optional[str] = None,
        sequence_length: int = 12,
        prediction_horizon: int = 12,
        normalize: bool = True,
    ):
        """
        Stage 5: Tạo sliding-window dataset cho T-GCN / DTC-STGCN.

        Sử dụng StandardScaler fit trên train set.
        Output shape: X [N_samples, seq_len, N_nodes, N_features].

        Normalization strategy (tối ưu):
            - NORMALIZE_COLS     → StandardScaler (fit trên train)
            - MINMAX_COLS        → MinMaxScaler (fit trên train)
            - Cyclic, binary     → không normalize
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 5: EXPORT FOR MODEL TRAINING")
        self.logger.info("=" * 60)

        df = self.npz_reader.read_features()
        if df is None or df.empty:
            self.logger.error("No features found to export")
            return

        self.logger.info(f"Loaded {len(df)} records, {len(df.columns)} columns")

        # ── Parse timestamps ─────────────────────────────────────────────────
        df = self._parse_timestamps(df)
        if df is None or df.empty:
            return

        # ── Select feature columns ────────────────────────────────────────────
        feature_cols = [
            col for col in FINAL_FEATURE_COLS
            if col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
            and df[col].nunique() > 1
        ]

        if not feature_cols:
            self.logger.error("No valid feature columns found.")
            return

        self.logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

        # ── Build 3D tensor [T, N, F] ─────────────────────────────────────────
        data_3d, segment_ids, num_timesteps = self._build_3d_tensor(
            df, feature_cols, sequence_length + prediction_horizon
        )
        if data_3d is None:
            return

        num_nodes, num_features = data_3d.shape[1], data_3d.shape[2]

        # ── Sliding window ────────────────────────────────────────────────────
        X, y = self._create_sliding_windows(data_3d, sequence_length, prediction_horizon)

        # ── Train / Val / Test split (temporal, NO shuffle) ───────────────────
        n = len(X)
        train_end = max(1, int(0.7 * n))
        val_end = train_end + max(1, int(0.15 * n))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        self.logger.info(
            f"Split → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        # ── Normalization ─────────────────────────────────────────────────────
        scaler_params = self._fit_and_normalize(
            X_train, X_val, X_test, y_train, y_val, y_test,
            feature_cols, normalize
        )
        X_train, y_train = scaler_params.pop("X_train"), scaler_params.pop("y_train")
        X_val, y_val = scaler_params.pop("X_val"), scaler_params.pop("y_val")
        X_test, y_test = scaler_params.pop("X_test"), scaler_params.pop("y_test")

        # ── Save ──────────────────────────────────────────────────────────────
        if output_name is None:
            output_name = f"model_ready_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        unique_dates = sorted(df["date"].unique()) if "date" in df.columns else []

        data_dict = {
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            "X_test":  X_test,  "y_test":  y_test,
            "segment_ids":   np.array(segment_ids),
            "feature_names": np.array(feature_cols),
            **scaler_params,
        }

        metadata = {
            "sequence_length":    sequence_length,
            "prediction_horizon": prediction_horizon,
            "num_features":       num_features,
            "num_nodes":          num_nodes,
            "num_timesteps":      num_timesteps,
            "train_size":         len(X_train),
            "val_size":           len(X_val),
            "test_size":          len(X_test),
            "normalized":         normalize,
            "date_range":         f"{unique_dates[0]} to {unique_dates[-1]}" if unique_dates else "",
            "feature_cols":       feature_cols,
        }

        self.npz_writer.write_batch(data_dict, output_name, metadata)

        self.logger.info(f"✅ Stage 5 complete: {output_name}")
        self.logger.info(
            f"   X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}"
        )

    # =========================================================================
    # STAGE 6 — Map Matching (MỚI — thay thế build_graph_structure)
    # =========================================================================

    def build_graph_with_map_matching(
        self,
        geometry: Dict,
        osm_output_name: str = "osm_graph",
        graph_output_name: str = "graph_structure",
        match_threshold_m: float = 50.0,
        force_rebuild_osm: bool = False,
    ):
        """
        Stage 6: Map-match TomTom features → OSM graph → graph_structure.npz.

        Thay thế hoàn toàn build_graph_structure() cũ (dùng distance_threshold).

        Args:
            geometry           : GeoJSON Polygon (cùng geometry với TomTom).
            osm_output_name    : Tên NPZ của OSM graph.
            graph_output_name  : Tên NPZ của matched graph output.
            match_threshold_m  : Ngưỡng khoảng cách match (m). Mặc định 50m.
            force_rebuild_osm  : Rebuild OSM graph dù có cache.
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 6: MAP MATCHING — TomTom → OSM Graph")
        self.logger.info("=" * 60)

        # 6.1 Load hoặc build OSM skeleton
        osm_graph_data = self.build_osm_skeleton(
            geometry=geometry,
            output_name=osm_output_name,
            force_rebuild=force_rebuild_osm,
        )

        if not osm_graph_data:
            self.logger.error("OSM graph không có. Dừng Stage 6.")
            return

        # 6.2 Load OSM NetworkX graph (để dùng ox.nearest_edges)
        try:
            import osmnx as ox
            coords = osm_graph_data["coordinates"]
            osm_node_ids = osm_graph_data["osm_node_ids"]
            edge_index = osm_graph_data["edge_index"]

            # Re-build NetworkX graph từ arrays (lightweight, không cần re-download OSM)
            G_nx = self._rebuild_nx_graph(osm_graph_data)

        except ImportError:
            self.logger.error("osmnx chưa cài. Cài: pip install osmnx")
            return

        # 6.3 Load TomTom features
        traffic_df = self.npz_reader.read_features()
        if traffic_df is None or traffic_df.empty:
            self.logger.error("No traffic features. Chạy Stage 4 trước.")
            return

        self.logger.info(
            f"Traffic data: {len(traffic_df)} records, "
            f"{traffic_df['segment_id'].nunique()} segments"
        )

        # 6.4 Map matching — tạo temporal features [N, T, F]
        matcher = TomTomOSMMapMatcher(
            osm_graph_data=osm_graph_data,
            match_threshold_m=match_threshold_m,
        )

        graph_output_dir = config.data.processed_dir / graph_output_name

        result = matcher.match_and_build_temporal_features(
            traffic_df=traffic_df,
            osm_networkx_graph=G_nx,
            output_name=graph_output_name,
            output_dir=graph_output_dir,
        )

        if not result:
            self.logger.error("Map matching failed.")
            return

        N = len(result.get("osm_node_ids", []))
        E = result.get("edge_index", np.zeros((2, 0))).shape[1]
        temporal_shape = result.get(
            "edge_features_temporal",
            result.get("node_features_temporal", np.array([]))
        ).shape

        self.logger.info("✅ Stage 6 complete — Matched subgraph built (Option B)")
        self.logger.info(f"   Subgraph nodes: {N} | Matched edges: {E}")
        self.logger.info(f"   Temporal features shape: {temporal_shape}")
        self.logger.info(
            f"   Coverage: {matcher._coverage_ratio:.1%} TomTom segments matched"
        )
        self.logger.info(
            "   Design: chỉ OSM edges có TomTom data — không có default giả"
        )

    # INTERNAL HELPERS
    def _parse_timestamps(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Parse date + time_set → global_timestamp.
        """
        if "time_set" not in df.columns:
            self.logger.error("Missing 'time_set' column")
            return None

        date_col = "date_from" if "date_from" in df.columns else "date_range"
        if date_col not in df.columns:
            self.logger.error("Missing date column (date_from / date_range)")
            return None

        def extract_date(s):
            s = str(s)
            if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
                return s
            m = re.search(r"\d{4}-\d{2}-\d{2}", s)
            return m.group(0) if m else None

        def extract_slot_info(ts):
            """
            Trả về (slot_index, block_id):
                slot_index: 0-11 cho sáng (07:00-09:45), 12-23 cho chiều (15:00-17:45)
                block_id  : 0 = sáng, 1 = chiều
            """
            m = re.search(r"Slot_(\d{4})", str(ts))
            if m:
                code = m.group(1)
                h, mn = int(code[:2]), int(code[2:])
                if 7 <= h < 10:
                    return (h - 7) * 4 + mn // 15, 0   # block sáng
                elif 15 <= h < 18:
                    return 12 + (h - 15) * 4 + mn // 15, 1  # block chiều
            return None, None

        df["date"] = df[date_col].apply(extract_date)
        slot_info = df["time_set"].apply(extract_slot_info)
        df["slot_index"] = slot_info.apply(lambda x: x[0])
        df["block_id"]   = slot_info.apply(lambda x: x[1])

        df = df.dropna(subset=["date", "slot_index"])
        df["slot_index"] = df["slot_index"].astype(int)
        df["block_id"]   = df["block_id"].astype(int)

        unique_dates = sorted(df["date"].unique())
        if not unique_dates:
            self.logger.error("No valid dates found.")
            return None

        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        df["day_index"] = df["date"].map(date_to_idx)

        # SLOTS_PER_DAY = 24 (12 sáng + 12 chiều)
        SLOTS_PER_DAY = 24
        df["global_timestamp"] = df["day_index"] * SLOTS_PER_DAY + df["slot_index"]

        df = df.sort_values(["segment_id", "global_timestamp"]).reset_index(drop=True)

        self.logger.info(
            f"Timestamps: {len(unique_dates)} days, "
            f"range {unique_dates[0]} → {unique_dates[-1]}, "
            f"blocks: sáng (slot 0-11) / chiều (slot 12-23)"
        )
        return df

    def _build_3d_tensor(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        min_timesteps: int,
    ):
        """Build 3D tensor [T, N, F] từ DataFrame."""
        all_segs = sorted(df["segment_id"].unique())
        ts_counts = df.groupby("segment_id")["global_timestamp"].count()
        num_days = df["date"].nunique()
        expected_ts = num_days * 24

        full_segs = [s for s in all_segs if ts_counts.get(s, 0) == expected_ts]
        candidate_segs = (
            full_segs
            if full_segs
            else [s for s in all_segs if ts_counts.get(s, 0) >= min_timesteps]
        )

        if not candidate_segs:
            self.logger.error(
                f"Không có segment nào có đủ {min_timesteps} timesteps. "
                f"Max available: {ts_counts.max() if len(ts_counts) else 0}"
            )
            return None, None, None

        num_timesteps = (
            expected_ts
            if full_segs
            else min(ts_counts[s] for s in candidate_segs)
        )

        segment_ids = sorted(candidate_segs)
        N = len(segment_ids)
        F = len(feature_cols)

        self.logger.info(
            f"Building 3D tensor: T={num_timesteps}, N={N}, F={F}"
        )

        data_3d = np.zeros((num_timesteps, N, F), dtype=np.float32)

        for node_idx, seg_id in enumerate(segment_ids):
            seg_df = df[df["segment_id"] == seg_id].sort_values("global_timestamp")
            vals = seg_df[feature_cols].values.astype(np.float32)

            if len(vals) > num_timesteps:
                vals = vals[:num_timesteps]
            elif len(vals) < num_timesteps:
                pad = np.tile(vals[-1:], (num_timesteps - len(vals), 1))
                vals = np.vstack([vals, pad])

            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            data_3d[:, node_idx, :] = vals

        return data_3d, segment_ids, num_timesteps

    def _create_sliding_windows(
        self,
        data_3d: np.ndarray,
        seq_len: int,
        pred_len: int,
    ):
        """
        Tạo sliding window dataset.
        """
        T = data_3d.shape[0]
        max_start = T - seq_len - pred_len + 1

        if max_start <= 0:
            self.logger.error(
                f"Không đủ timesteps ({T}) để tạo window "
                f"(seq={seq_len} + pred={pred_len})."
            )
            return np.array([]), np.array([])

        SLOTS_PER_DAY = 24
        MORNING_LAST  = 11   # slot cuối block sáng (07:00-09:45)
        EVENING_FIRST = 12   # slot đầu block chiều (15:00-17:45)

        def _crosses_block_gap(start: int, length: int) -> bool:
            """
            Kiểm tra xem window [start, start+length) có cắt qua ranh giới
            block sáng → chiều trong bất kỳ ngày nào không.
            """
            for t in range(start, start + length - 1):
                slot_in_day_t   = t % SLOTS_PER_DAY
                slot_in_day_t1  = (t + 1) % SLOTS_PER_DAY
                # Chuyển từ slot cuối sáng (11) sang slot đầu chiều (12)
                # trong CÙNG ngày → là gap thực
                day_t  = t // SLOTS_PER_DAY
                day_t1 = (t + 1) // SLOTS_PER_DAY
                if (day_t == day_t1
                        and slot_in_day_t == MORNING_LAST
                        and slot_in_day_t1 == EVENING_FIRST):
                    return True
            return False

        X_list, y_list = [], []
        skipped = 0
        window_len = seq_len + pred_len

        for i in range(max_start):
            if _crosses_block_gap(i, window_len):
                skipped += 1
                continue
            X_list.append(data_3d[i: i + seq_len])
            y_list.append(data_3d[i + seq_len: i + seq_len + pred_len])

        if skipped > 0:
            self.logger.info(
                f"Sliding windows: {skipped} windows bỏ qua (cắt qua block gap sáng/chiều)"
            )

        if not X_list:
            self.logger.error("Không có window hợp lệ sau khi lọc block gap.")
            return np.array([]), np.array([])

        X = np.array(X_list, dtype=np.float32)  # [N_samples, seq_len, N_nodes, F]
        y = np.array(y_list, dtype=np.float32)  # [N_samples, pred_len, N_nodes, F]

        self.logger.info(f"Sliding windows: X={X.shape}, y={y.shape}")
        return X, y

    def _fit_and_normalize(
        self,
        X_train, X_val, X_test, y_train, y_val, y_test,
        feature_cols: List[str],
        normalize: bool,
    ) -> Dict[str, Any]:
        """
        Fit scalers trên train set, transform tất cả splits.

        Trả về dict gồm normalized arrays + scaler params.
        """
        n_features = X_train.shape[-1]

        # Map feature name → scaler type
        standard_idx = [
            i for i, c in enumerate(feature_cols) if c in NORMALIZE_COLS
        ]
        minmax_idx = [
            i for i, c in enumerate(feature_cols) if c in MINMAX_COLS
        ]

        result = {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
        }

        if not normalize:
            # Fit scaler nhưng không transform (lưu params để dùng sau)
            X_tr_2d = X_train.reshape(-1, n_features)
            ss = StandardScaler().fit(X_tr_2d[:, standard_idx] if standard_idx else X_tr_2d)
            result["scaler_mean"] = ss.mean_.astype(np.float32) if standard_idx else np.zeros(n_features, dtype=np.float32)
            result["scaler_scale"] = ss.scale_.astype(np.float32) if standard_idx else np.ones(n_features, dtype=np.float32)
            result["standard_feature_idx"] = np.array(standard_idx, dtype=np.int32)
            result["minmax_feature_idx"] = np.array(minmax_idx, dtype=np.int32)
            self.logger.info("Scaler params saved (normalize=False, raw data exported).")
            return result

        X_tr_2d = X_train.reshape(-1, n_features)

        # StandardScaler cho continuous speed/time features
        ss_mean = np.zeros(n_features, dtype=np.float32)
        ss_scale = np.ones(n_features, dtype=np.float32)

        if standard_idx:
            ss = StandardScaler()
            ss.fit(X_tr_2d[:, standard_idx])
            ss_mean[standard_idx] = ss.mean_.astype(np.float32)
            ss_scale[standard_idx] = ss.scale_.astype(np.float32)

        # MinMaxScaler cho distance, speed_limit, sample_size
        mm_min = np.zeros(n_features, dtype=np.float32)
        mm_scale = np.ones(n_features, dtype=np.float32)

        if minmax_idx:
            mm = MinMaxScaler()
            mm.fit(X_tr_2d[:, minmax_idx])
            mm_min[minmax_idx] = mm.data_min_.astype(np.float32)
            mm_scale[minmax_idx] = (mm.data_max_ - mm.data_min_ + 1e-8).astype(np.float32)

        def _transform(arr: np.ndarray) -> np.ndarray:
            s0, s1, s2, s3 = arr.shape
            a = arr.reshape(-1, s3).copy()
            if standard_idx:
                a[:, standard_idx] = (a[:, standard_idx] - ss_mean[standard_idx]) / ss_scale[standard_idx]
            if minmax_idx:
                a[:, minmax_idx] = (a[:, minmax_idx] - mm_min[minmax_idx]) / mm_scale[minmax_idx]
            return a.reshape(s0, s1, s2, s3).astype(np.float32)

        result["X_train"] = _transform(X_train)
        result["X_val"]   = _transform(X_val)
        result["X_test"]  = _transform(X_test)
        result["y_train"] = _transform(y_train)
        result["y_val"]   = _transform(y_val)
        result["y_test"]  = _transform(y_test)
        result["scaler_mean"]  = ss_mean
        result["scaler_scale"] = ss_scale
        result["mm_min"]       = mm_min
        result["mm_scale"]     = mm_scale
        result["standard_feature_idx"] = np.array(standard_idx, dtype=np.int32)
        result["minmax_feature_idx"]   = np.array(minmax_idx, dtype=np.int32)

        self.logger.info(
            f"✅ Normalization done. "
            f"StandardScaler: {len(standard_idx)} features. "
            f"MinMaxScaler: {len(minmax_idx)} features."
        )
        return result

    @staticmethod
    def _rebuild_nx_graph(osm_data: Dict[str, np.ndarray]) -> "nx.MultiDiGraph":
        """
        Tái tạo NetworkX graph từ OSM arrays để dùng ox.nearest_edges.
        Nhẹ hơn nhiều so với re-download OSM.

        BUG FIX: thêm geometry (LineString) vào mỗi edge để ox.nearest_edges
        tính khoảng cách chính xác đến đoạn thẳng thay vì chỉ đến node gần nhất.
        Nếu không có geometry, nearest_edges dùng khoảng cách tới node trung điểm
        → sai lệch lớn với segments dài, gây coverage = 0.
        """
        import networkx as nx
        try:
            from shapely.geometry import LineString
            _has_shapely = True
        except ImportError:
            _has_shapely = False

        G = nx.MultiDiGraph()
        G.graph["crs"] = "epsg:4326"
        node_ids = osm_data["osm_node_ids"]
        coords = osm_data["coordinates"]          # [N, 2] (lat, lon)
        edge_index = osm_data["edge_index"]
        edge_lengths = osm_data.get("edge_lengths", np.zeros(edge_index.shape[1]))

        # Add nodes — x=lon, y=lat (OSMnx convention)
        for idx, nid in enumerate(node_ids):
            G.add_node(int(nid), y=float(coords[idx, 0]), x=float(coords[idx, 1]))

        # Add edges với geometry để ox.nearest_edges tính dist đúng.
        # BUG FIX (cũ): dùng key=(min,max) để dedup → mất cung ngược chiều.
        # FIX: thêm tất cả cung, dùng e_idx làm key.
        for e_idx in range(edge_index.shape[1]):
            u_idx = edge_index[0, e_idx]
            v_idx = edge_index[1, e_idx]
            u_nid = int(node_ids[u_idx])
            v_nid = int(node_ids[v_idx])
            length = float(edge_lengths[e_idx]) if e_idx < len(edge_lengths) else 0.0

            edge_attrs: Dict = {"key": e_idx, "length": length}

            # Tạo LineString từ tọa độ u→v để nearest_edges dùng projected distance.
            # OSMnx dùng geometry.coords cho khoảng cách đến đường thẳng.
            if _has_shapely:
                u_lon = float(coords[u_idx, 1])
                u_lat = float(coords[u_idx, 0])
                v_lon = float(coords[v_idx, 1])
                v_lat = float(coords[v_idx, 0])
                edge_attrs["geometry"] = LineString(
                    [(u_lon, u_lat), (v_lon, v_lat)]
                )

            G.add_edge(u_nid, v_nid, **edge_attrs)

        return G

    # =========================================================================
    # FULL PIPELINE ENTRY POINT
    # =========================================================================

    def run_full_pipeline(
        self,
        geometry: Dict,
        start_date: str,
        use_31_days: bool = True,
        job_name: str = "Traffic Analysis",
        match_threshold_m: float = 50.0,
    ):
        """Chạy toàn bộ pipeline từ collection đến graph building."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING FULL PIPELINE (OSM + TomTom Hybrid)")
        self.logger.info("=" * 60)

        # Stage 2b có thể chạy trước hoặc song song với 2a
        self.build_osm_skeleton(geometry=geometry)

        if use_31_days:
            job_ids = self.run_31_days_collection(
                geometry=geometry, start_date=start_date
            )
            if not job_ids:
                self.logger.error("Pipeline failed at Stage 1")
                return
            self.run_streaming_ingestion_from_jobs(job_ids)
        else:
            self.logger.error("Chỉ hỗ trợ 31-day mode trong.")
            return

        self.run_validation_processing()
        self.run_feature_extraction()
        self.export_for_model_training()
        self.build_graph_with_map_matching(
            geometry=geometry,
            match_threshold_m=match_threshold_m,
        )

        self.logger.info("=" * 60)
        self.logger.info("✅ FULL PIPELINE COMPLETE")
        self.logger.info("=" * 60)