# scripts/run_pipeline.py
"""
Script chạy pipeline xử lý dữ liệu theo Kappa Architecture
Lưu trữ vào NPZ files thay vì database
"""

from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_processing.pipeline_npz import TrafficDataPipelineNPZ
from data_processing.storage.npz_storage import NPZReader
from utils.config import config

def main():
    """Main pipeline execution"""
    
    # Khởi tạo pipeline
    pipeline = TrafficDataPipelineNPZ()
    
    # Định nghĩa geometry cho District 1, HCMC
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [106.6850, 10.7850],  # Southwest corner
            [106.7100, 10.7850],  # Southeast corner
            [106.7100, 10.8000],  # Northeast corner
            [106.6850, 10.8000],  # Northwest corner
            [106.6850, 10.7700]   # Close the polygon
        ]]
    }
    
    # Chạy full pipeline
    # pipeline.run_full_pipeline(
    #     geometry=geometry,
    #     date_from="2024-08-01",
    #     date_to="2024-08-31",
    #     job_name="District 1 Traffic August 2024"
    # )
    # Stage 2: Ingestion
    pipeline.run_streaming_ingestion("8375763")
    
    # Stage 3: Validation
    pipeline.run_validation_processing()
    
    # Stage 4: Feature Extraction
    pipeline.run_feature_extraction()
    
    # Stage 5: Export for training
    pipeline.export_for_model_training()
    
    # Stage 6: Build graph
    pipeline.build_graph_structure()
    
    pipeline.logger.info("=" * 60)
    pipeline.logger.info("✅ FULL PIPELINE COMPLETE")
    pipeline.logger.info("=" * 60)
    
    # Kiểm tra kết quả
    reader = NPZReader()
    datasets = reader.list_datasets()
    print("\n" + "="*60)
    print("AVAILABLE DATASETS:")
    print("="*60)
    for dataset in datasets:
        info = reader.get_dataset_info(dataset)
        print(f"\n📦 {dataset}")
        print(f"   Files: {info.get('num_files', 0)}")
        print(f"   Size: {info.get('total_size_mb', 0):.2f} MB")
        if 'metadata' in info:
            print(f"   Metadata: {info['metadata']}")

if __name__ == "__main__":
    main()