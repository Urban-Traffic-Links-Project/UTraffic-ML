"""
Run traffic data preprocessing pipeline for District 1 (Quan 1), HCMC
"""

from data_processing.pipeline import TrafficDataPipeline
from data_processing.storage.parquet_writer import ParquetReader

def main():
    pipeline = TrafficDataPipeline()

    # Geometry: Quận 1
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [106.6816889, 10.7535894],
                [106.7151954, 10.7535894],
                [106.7151954, 10.7969493],
                [106.6816889, 10.7969493],
                [106.6816889, 10.7535894]
            ]
        ]
    }

    # Thời gian
    date_from = "2024-08-01"
    date_to = "2024-08-31"
    job_name = "District 1 Traffic August 2024"

    print("\n" + "=" * 70)
    print("TRAFFIC DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    print("Area: District 1 (Quan 1), Ho Chi Minh City")
    print(f"Period: {date_from} → {date_to}")
    print(f"Job name: {job_name}")
    print("=" * 70 + "\n")

    # Chạy pipeline đầy đủ
    pipeline.run_full_pipeline(
        geometry=geometry,
        date_from=date_from,
        date_to=date_to,
        job_name=job_name
    )

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)

    reader = ParquetReader()
    tables = reader.list_tables()

    print(f"\nAvailable tables: {tables}")

    if 'traffic_features' in tables:
        df = reader.read_features()
        print("\nTraffic features dataset:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print("\nSample rows:")
        print(df.head())

        print("\nFeature columns:")
        for col in sorted(df.columns):
            print(f"  - {col}")


if __name__ == "__main__":
    main()
