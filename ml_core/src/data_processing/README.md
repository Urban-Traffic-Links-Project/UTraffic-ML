# Traffic Data Preprocessing Pipeline

Hệ thống tiền xử lý dữ liệu giao thông theo kiến trúc Kappa Architecture, được thiết kế để thu thập, validate, làm sạch, trích xuất features và lưu trữ dữ liệu giao thông từ TomTom API.

## 🏗️ Kiến trúc

Pipeline tuân theo **Kappa Architecture** với 4 stages chính:

```
Stage 1: Data Collection (TomTom API)
    ↓
Stage 2: Streaming Ingestion (Kafka: traffic.raw)
    ↓
Stage 3: Validation (Kafka: traffic.validated)
    ↓
Stage 4: Feature Extraction & Storage (Parquet)
```

## 📁 Cấu trúc thư mục

```
src/
├── data_processing/
│   ├── collectors/
│   │   └── tomtom_collector.py      # Thu thập dữ liệu từ TomTom API
│   ├── streaming/
│   │   ├── kafka_producer.py        # Gửi dữ liệu vào Kafka
│   │   └── kafka_consumer.py        # Nhận và xử lý từ Kafka
│   ├── preprocessors/
│   │   ├── data_validator.py        # Validate dữ liệu
│   │   ├── data_cleaner.py          # Làm sạch dữ liệu
│   │   ├── feature_extractor.py     # Trích xuất features
│   │   └── data_normalizer.py       # Chuẩn hóa dữ liệu
│   ├── storage/
│   │   └── parquet_writer.py        # Lưu/đọc Parquet files
│   └── pipeline.py                   # Main pipeline orchestrator
├── utils/
│   ├── config.py                     # Configuration management
│   └── logger.py                     # Logging utilities
data/
├── raw/                              # Dữ liệu thô từ TomTom
│   └── tomtom_stats_frc5/
└── processed/                        # Dữ liệu đã xử lý
    └── parquet/                      # Parquet datasets
        ├── traffic_features/         # Features chính
        ├── raw_traffic_data/         # Raw data backup
        └── validated_traffic_data/   # Validated data
```

## 🚀 Cài đặt

### 1. Dependencies

```bash
pip install -r requirements.txt
```

### 2. Kafka Setup

Cần có Kafka cluster đang chạy. Sử dụng Docker Compose:

```bash
docker-compose up -d zookeeper kafka
```

### 3. Environment Variables

Tạo file `.env`:

```bash
# TomTom API
TOMTOM_TRAFFIC_STATS_API_KEY=your_api_key_here

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Logging
LOG_LEVEL=INFO
```

## 📊 Sử dụng

### Run Full Pipeline

```python
from src.data_processing.pipeline import TrafficDataPipeline

pipeline = TrafficDataPipeline()

# Định nghĩa khu vực
geometry = {
    "type": "Polygon",
    "coordinates": [
        [
            [106.730, 10.870],
            [106.770, 10.870],
            [106.770, 10.910],
            [106.730, 10.910],
            [106.730, 10.870]
        ]
    ]
}

# Chạy pipeline
pipeline.run_full_pipeline(
    geometry=geometry,
    date_from="2024-08-01",
    date_to="2024-08-31",
    job_name="Thu Duc Traffic Analysis"
)
```

### Run Separate Stages

```python
# Stage 1: Collection
job_id = pipeline.run_batch_collection(geometry, "2024-08-01", "2024-08-31")

# Stage 2: Ingestion to Kafka
pipeline.run_streaming_ingestion(job_id)

# Stage 3: Validation
pipeline.run_validation_processing()

# Stage 4: Feature Extraction
pipeline.run_feature_extraction()
```

### Read Processed Data

```python
from src.data_processing.storage.parquet_writer import ParquetReader

reader = ParquetReader()

# Đọc toàn bộ features
df = reader.read_features()

# Đọc với filters
df = reader.read_features(
    filters=[('date', '=', '2024-08-15')],
    columns=['segment_id', 'average_speed', 'timestamp']
)
```

## 🔧 Configuration

Cấu hình được quản lý trong `src/utils/config.py`:

```python
from src.utils.config import config

# Kafka settings
config.kafka.bootstrap_servers = "localhost:9092"
config.kafka.raw_topic = "traffic.raw"

# Data quality thresholds
config.data.missing_threshold = 0.20
config.data.z_score_threshold = 3.0

# Feature extraction
config.features.time_windows = [12, 36, 144]  # 1h, 3h, 12h
```

## 📈 Data Processing Steps

### 1. Data Validation
- Kiểm tra cấu trúc dữ liệu
- Validate giá trị (speed, travel time, coordinates)
- Loại bỏ records không hợp lệ

### 2. Data Cleaning
- Xử lý missing values (interpolation, median filling)
- Phát hiện outliers (Z-score + IQR methods)
- Remove duplicates và invalid rows

### 3. Feature Extraction
- **Speed features**: relative_speed, speed_reduction_ratio
- **Temporal features**: hour_sin/cos, day_of_week, is_peak
- **Congestion features**: congestion_index, travel_time_ratio
- **Dynamic features**: speed derivatives, moving averages, volatility
- **Statistical features**: z-score, percentile, coefficient of variation

### 4. Normalization
- Standard scaling (Z-score)
- Min-Max scaling
- Robust scaling (IQR-based)

## 🗂️ Parquet Storage

Dữ liệu được lưu trong Parquet format với partitioning:

```
traffic_features/
├── date=2024-08-01/
│   ├── part-0.parquet
│   └── part-1.parquet
├── date=2024-08-02/
│   └── part-0.parquet
...
```

**Advantages:**
- Columnar storage → fast queries
- Compression → reduced storage
- Partitioning → efficient filtering
- Schema evolution support

## 📝 Logging

Logs được lưu tại `logs/pipeline.log`:

```
2025-12-14 10:30:00 - TrafficDataPipeline - INFO - ✅ Pipeline initialized
2025-12-14 10:30:05 - TomTomTrafficDataCollector - INFO - ✅ Job created successfully! Job ID: 7839872
2025-12-14 10:35:00 - DataCleaner - INFO - Cleaning data: 15000 rows
2025-12-14 10:35:10 - FeatureExtractor - INFO - Feature extraction complete. Total features: 45
```

## 🔍 Monitoring

### Kafka Consumer Lag

```bash
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --describe --group traffic-processor
```

### Parquet Data Statistics

```python
from src.data_processing.storage.parquet_writer import ParquetReader

reader = ParquetReader()
info = reader.get_table_info('traffic_features')
print(info)
```

## 🐛 Troubleshooting

### Kafka Connection Error
```
Error: Failed to connect to Kafka
```
**Solution**: Đảm bảo Kafka đang chạy và `KAFKA_BOOTSTRAP_SERVERS` đúng.

### TomTom API Rate Limit
```
Error: API Error 429
```
**Solution**: Giảm số lượng requests hoặc chờ rate limit reset.

### Missing Values Warning
```
Warning: Column average_speed has 25% missing data
```
**Solution**: Kiểm tra nguồn dữ liệu hoặc adjust `missing_threshold` trong config.

## 📚 Next Steps

Sau khi data được xử lý và lưu trong Parquet:

1. **Correlation Analysis**: Phân tích tương quan giữa các segments
2. **Granger Causality**: Xác định quan hệ nhân quả
3. **Model Training**: Huấn luyện Dynamic GCN models
4. **Visualization**: Tạo dashboards và reports

## 🤝 Contributing

Xem chi tiết về cấu trúc code và best practices trong documentation.

## 📄 License

This project is part of the HCMC Traffic Analysis thesis at HCMUT.