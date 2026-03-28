# src/data_processing/run_pipeline_quan1_npz.py
from pathlib import Path
import re
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_processing.pipeline_npz import TrafficDataPipelineNPZ
from data_processing.storage.npz_storage import NPZReader
from data_processing.graph.osm_graph_builder import OSMGraphBuilder
from utils.config import config

# Chọn chế độ chạy:
#   "from_stage_5"  ← BẮT ĐẦU TỪ ĐÂY nếu đã có traffic_features_*.npz
#   "from_stage_2b" ← Nếu muốn re-extract features với extractor
#   "graph_only"    ← Chỉ build graph với map matching
#   "osm_only"      ← Chỉ build OSM skeleton (để test)
#   "full_pipeline" ← Chạy toàn bộ từ đầu
MODE = "graph_only"

START_DATE = "2024-08-01"

# Map matching threshold (m) — segment TomTom cách OSM edge > threshold này sẽ bị bỏ qua
MATCH_THRESHOLD_M = 50.0

# Sequence length và prediction horizon cho sliding window
SEQUENCE_LENGTH = 12     # 12 slots × 15 phút = 3 giờ
PREDICTION_HORIZON = 12  # dự báo 3 giờ tiếp theo

def get_geometry_quan1():
    """
    GeoJSON Polygon cho Quận 1, HCMC (EPSG:4326).
    Ring đóng: điểm đầu = điểm cuối. Thứ tự [longitude, latitude].
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


def job_ids_from_raw_dir():
    """Lấy danh sách job_id từ file job_*_results.json đã có."""
    raw_dir = config.data.raw_dir / "tomtom_stats_frc5"
    if not raw_dir.exists():
        return []
    pattern = re.compile(r"job_(\d+)_results\.json")
    ids = []
    for p in sorted(raw_dir.glob("job_*_results.json")):
        m = pattern.match(p.name)
        if m:
            ids.append(m.group(1))
    return ids

def main():
    pipeline = TrafficDataPipelineNPZ()
    geometry = get_geometry_quan1()

    pipeline.logger.info("=" * 60)
    pipeline.logger.info(f"PIPELINE — MODE: {MODE}")
    pipeline.logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # MODE: from_stage_5
    # Dùng khi: traffic_features_*.npz đã có (Stage 4 done)
    #           Cần build OSM skeleton + chạy export + map matching
    #
    # BƯỚC NÀY SẼ:
    #   1. Build OSM skeleton (Stage 2b) — lần đầu mất ~2 phút download OSM
    #   2. Export model training data với feature_extractor (Stage 5)
    #   3. Map matching TomTom → OSM → graph_structure.npz (Stage 6)
    # -------------------------------------------------------------------------
    if MODE == "from_stage_5":
        pipeline.logger.info("▶ Starting from Stage 5 (traffic_features đã có)")
        pipeline.logger.info("  Bao gồm: Stage 2b (OSM) + Stage 5 (export) + Stage 6 (map matching)")

        # Stage 2b: Build OSM skeleton (có cache, chỉ download lần đầu)
        pipeline.logger.info("\n--- Stage 2b: Build OSM Skeleton ---")
        pipeline.build_osm_skeleton(geometry=geometry, force_rebuild=False)

        # Stage 5: Export với feature mới (~28 features)
        # LƯU Ý: Nếu traffic_features_*.npz đã có từ extractor cũ (57 features),
        # Stage 5 vẫn chạy được vì chỉ đọc FINAL_FEATURE_COLS có trong file.
        # Features không tồn tại sẽ bị bỏ qua (không lỗi).
        pipeline.logger.info("\n--- Stage 5: Export for Model Training ---")
        pipeline.export_for_model_training(
            sequence_length=SEQUENCE_LENGTH,
            prediction_horizon=PREDICTION_HORIZON,
            normalize=True,
        )

        # Stage 6: Map matching → graph_structure.npz
        pipeline.logger.info("\n--- Stage 6: Map Matching TomTom → OSM ---")
        pipeline.build_graph_with_map_matching(
            geometry=geometry,
            match_threshold_m=MATCH_THRESHOLD_M,
        )

    # -------------------------------------------------------------------------
    # MODE: from_stage_2b
    # Dùng khi: Muốn re-extract features hoàn toàn với FeatureExtractor
    #           (traffic_features_*.npz cũ có 57 features, muốn rebuild sạch với 28)
    #
    # BƯỚC NÀY SẼ:
    #   1. Build OSM skeleton (Stage 2b)
    #   2. Re-ingest từ job files đã có → Kafka (Stage 2a)
    #   3. Validation (Stage 3)
    #   4. Feature extraction → traffic_features mới (Stage 4)
    #   5. Export (Stage 5)
    #   6. Map matching (Stage 6)
    # -------------------------------------------------------------------------
    elif MODE == "from_stage_2b":
        pipeline.logger.info("▶ Starting from Stage 2b (re-extract features với extractor)")

        job_ids = job_ids_from_raw_dir()
        if not job_ids:
            pipeline.logger.error(
                "❌ Không tìm thấy job_*_results.json. "
                "Cần chạy Stage 1 (TomTom collection) trước."
            )
            return

        pipeline.logger.info(f"  Found {len(job_ids)} job files.")

        # Stage 2b: OSM Skeleton (song song, độc lập với Kafka)
        pipeline.logger.info("\n--- Stage 2b: Build OSM Skeleton ---")
        pipeline.build_osm_skeleton(geometry=geometry, force_rebuild=False)

        # Stage 2a: Re-ingest từ raw JSON → Kafka
        pipeline.logger.info("\n--- Stage 2a: Re-ingest to Kafka ---")
        pipeline.run_streaming_ingestion_from_jobs(job_ids)

        # Stage 3: Validation
        pipeline.logger.info("\n--- Stage 3: Validation ---")
        pipeline.run_validation_processing()

        # Stage 4: Feature extraction
        pipeline.logger.info("\n--- Stage 4: Feature Extraction (~28 features) ---")
        pipeline.run_feature_extraction()

        # Stage 5: Export
        pipeline.logger.info("\n--- Stage 5: Export for Model Training ---")
        pipeline.export_for_model_training(
            sequence_length=SEQUENCE_LENGTH,
            prediction_horizon=PREDICTION_HORIZON,
            normalize=True,
        )

        # Stage 6: Map matching
        pipeline.logger.info("\n--- Stage 6: Map Matching TomTom → OSM ---")
        pipeline.build_graph_with_map_matching(
            geometry=geometry,
            match_threshold_m=MATCH_THRESHOLD_M,
        )

    # -------------------------------------------------------------------------
    # MODE: graph_only
    # Dùng khi: OSM skeleton và traffic_features đã có, chỉ cần build graph lại
    # -------------------------------------------------------------------------
    elif MODE == "graph_only":
        pipeline.logger.info("▶ Graph only — Stage 6: Map Matching")

        pipeline.build_osm_skeleton(geometry=geometry, force_rebuild=False)
        pipeline.build_graph_with_map_matching(
            geometry=geometry,
            match_threshold_m=MATCH_THRESHOLD_M,
        )

    # -------------------------------------------------------------------------
    # MODE: osm_only
    # Dùng khi: Chỉ muốn test build OSM skeleton và xem kết quả
    # -------------------------------------------------------------------------
    elif MODE == "osm_only":
        pipeline.logger.info("▶ OSM only — Stage 2b: Build OSM Skeleton")

        osm_data = pipeline.build_osm_skeleton(geometry=geometry, force_rebuild=False)

        if osm_data:
            N = len(osm_data["osm_node_ids"])
            E = osm_data["edge_index"].shape[1]
            coords = osm_data["coordinates"]
            pipeline.logger.info(f"\n📊 OSM Graph Summary:")
            pipeline.logger.info(f"   Nodes (ngã tư): {N}")
            pipeline.logger.info(f"   Directed edges: {E}  |  Undirected: {E // 2}")
            pipeline.logger.info(
                f"   Lat range: [{coords[:, 0].min():.5f}, {coords[:, 0].max():.5f}]"
            )
            pipeline.logger.info(
                f"   Lon range: [{coords[:, 1].min():.5f}, {coords[:, 1].max():.5f}]"
            )
            pipeline.logger.info(f"   Avg degree: {osm_data['adjacency_matrix'].sum() / N:.2f}")

    # -------------------------------------------------------------------------
    # MODE: full_pipeline
    # Dùng khi: Chạy toàn bộ từ đầu (cần TomTom API key, mất nhiều thời gian)
    # -------------------------------------------------------------------------
    elif MODE == "full_pipeline":
        pipeline.logger.info("▶ Full pipeline từ đầu")
        pipeline.run_full_pipeline(
            geometry=geometry,
            start_date=START_DATE,
            use_31_days=True,
            match_threshold_m=MATCH_THRESHOLD_M,
        )

    else:
        pipeline.logger.error(f"Unknown MODE: {MODE}")
        return

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    pipeline.logger.info("\n" + "=" * 60)
    pipeline.logger.info("✅ PIPELINE COMPLETE")
    pipeline.logger.info("=" * 60)

    reader = NPZReader()
    datasets = reader.list_datasets()

    print("\n" + "=" * 60)
    print("AVAILABLE DATASETS:")
    print("=" * 60)

    for dataset in datasets:
        info = reader.get_dataset_info(dataset)
        print(f"\n📦 {dataset}")
        print(f"   Files : {info.get('num_files', 0)}")
        print(f"   Size  : {info.get('total_size_mb', 0):.2f} MB")

        meta = info.get("metadata", {})
        if isinstance(meta, dict):
            for k in ["num_nodes", "num_edges_directed", "num_timesteps",
                      "num_features", "train_size", "topology", "date_range"]:
                if k in meta:
                    print(f"   {k}: {meta[k]}")


if __name__ == "__main__":
    main()