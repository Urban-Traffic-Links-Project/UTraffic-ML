# src/data_processing/run_pipeline_quan1_npz.py
"""
Script chạy pipeline xử lý dữ liệu theo Kappa Architecture.
Thu thập 31 ngày (mỗi ngày 1 job), lưu NPZ.
"""

from pathlib import Path
import re
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_processing.pipeline_npz import TrafficDataPipelineNPZ
from data_processing.storage.npz_storage import NPZReader
from utils.config import config


def get_geometry_quan1():
    """
    Geometry cho Quận 1, HCMC (GeoJSON Polygon, EPSG:4326).
    Ring phải đóng: điểm đầu = điểm cuối. Thứ tự [longitude, latitude].
    """
    # SW -> SE -> NE -> NW -> SW (đóng polygon)
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
    """
    Lấy danh sách job_id từ các file job_*_results.json trong data/raw/tomtom_stats_frc5.
    Dùng khi đã thu 31 ngày và chỉ cần chạy từ Stage 2 (ingestion).
    Thứ tự theo tên file (job_id).
    """
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

    # --- Chế độ 1: Full pipeline 31 ngày (thu thập + ingestion + validation + features + export + graph)
    RUN_FULL_31_DAYS = False
    START_DATE = "2024-08-01"

    if RUN_FULL_31_DAYS:
        pipeline.run_full_pipeline(
            geometry=geometry,
            start_date=START_DATE,
            use_31_days=True,
            job_name="District 1 Traffic 31 Days",
        )
    else:
        # --- Chế độ 2: Chỉ chạy từ Stage 2 khi đã có sẵn file job_*_results.json (31 ngày)
        job_ids = job_ids_from_raw_dir()
        if not job_ids:
            pipeline.logger.warning("No job_*_results.json found. Run full pipeline first.")
            return
        pipeline.logger.info(f"Found {len(job_ids)} job files. Running from Stage 2.")
        pipeline.run_streaming_ingestion_from_jobs(job_ids)
        pipeline.run_validation_processing()
        pipeline.run_feature_extraction()
        pipeline.export_for_model_training()
        pipeline.build_graph_structure()
        pipeline.logger.info("=" * 60)
        pipeline.logger.info("✅ PIPELINE (FROM INGESTION) COMPLETE")
        pipeline.logger.info("=" * 60)

    # In danh sách dataset NPZ
    reader = NPZReader()
    datasets = reader.list_datasets()
    print("\n" + "=" * 60)
    print("AVAILABLE DATASETS:")
    print("=" * 60)
    for dataset in datasets:
        info = reader.get_dataset_info(dataset)
        print(f"\n📦 {dataset}")
        print(f"   Files: {info.get('num_files', 0)}")
        print(f"   Size: {info.get('total_size_mb', 0):.2f} MB")
        if "metadata" in info:
            print(f"   Metadata: {info['metadata']}")


if __name__ == "__main__":
    main()
