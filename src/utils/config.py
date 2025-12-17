# src/utils/config.py
import os
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

@dataclass
class KafkaConfig:
    """Kafka configuration"""
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    raw_topic: str = "traffic.raw"
    validated_topic: str = "traffic.validated"
    features_topic: str = "traffic.features"
    consumer_group: str = "traffic-processor"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    max_poll_records: int = 500

@dataclass
class TomTomConfig:
    """TomTom API configuration"""
    api_key: str = os.getenv("TOMTOM_TRAFFIC_STATS_API_KEY", "")
    base_url: str = "https://api.tomtom.com/traffic/trafficstats"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5

@dataclass
class DataConfig:
    """Data processing configuration"""
    # Paths
    project_dir: Path = Path(os.getcwd()).parent.parent
    data_dir: Path = project_dir / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    parquet_dir: Path = processed_dir / "parquet"

    # Quality thresholds
    missing_threshold: float = 0.20  # Max 20% missing data
    confidence_threshold: float = 0.7  # Min confidence score

    # Outlier detection
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5

    # Speed limits
    min_speed: float = 0.0  # km/h
    max_speed: float = 120.0  # km/h

    # Time windows
    interpolation_gap_max: int = 3  # Max time steps for interpolation

    def __post_init__(self):
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.parquet_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class FeatureConfig:
    """Feature extraction configuration"""
    # Temporal features
    time_windows: List[int] = field(default_factory=lambda: [12, 36, 144])  # [12, 36, 144] steps
    
    # Spatial features
    distance_threshold: float = 5.0  # Chỉ xem xét các đoạn đường trong bán kính km
    connectivity_radius: float = 2.0  # Xác định láng giềng trực tiếp km
    
    # Speed features
    use_relative_speed: bool = True
    use_speed_variance: bool = True
    
    # Congestion features
    use_congestion_index: bool = True
    use_travel_time_ratio: bool = True

@dataclass
class Config:
    """Main configuration"""
    kafka: KafkaConfig = None
    tomtom: TomTomConfig = None
    data: DataConfig = None
    features: FeatureConfig = None
    
    # Processing
    batch_size: int = 1000
    num_workers: int = 4
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = "logs/pipeline.log"
    
    def __post_init__(self):
        if self.kafka is None:
            self.kafka = KafkaConfig()
        if self.tomtom is None:
            self.tomtom = TomTomConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.features is None:
            self.features = FeatureConfig()
        
        # Create log directory
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

# Global config instance
config = Config()
