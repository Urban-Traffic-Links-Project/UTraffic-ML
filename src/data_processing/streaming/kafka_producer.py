# src/data_processing/streaming/kafka_producer.py
import json
from typing import Dict, Any, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError

from ...utils.config import config
from ...utils.logger import LoggerMixin

class TrafficDataProducer(LoggerMixin):
    """
    Kafka Producer để gửi dữ liệu giao thông vào Kafka topics
    """
    
    def __init__(self, bootstrap_servers: Optional[str] = None):
        self.bootstrap_servers = bootstrap_servers or config.kafka.bootstrap_servers
        self.producer = None
        self._connect()
    
    def _connect(self):
        """Khởi tạo Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas
                retries=3,
                max_in_flight_requests_per_connection=1,  # Ensure ordering
                compression_type='gzip'
            )
            self.logger.info(f"✅ Connected to Kafka: {self.bootstrap_servers}")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Kafka: {e}")
            raise
    
    def send_raw_data(self, data: Dict[str, Any], key: Optional[str] = None):
        """
        Gửi dữ liệu thô vào topic traffic.raw
        
        Args:
            data: Dữ liệu cần gửi
            key: Message key (segment_id, job_id, etc.)
        """
        return self._send(config.kafka.raw_topic, data, key)
    
    def send_validated_data(self, data: Dict[str, Any], key: Optional[str] = None):
        """Gửi dữ liệu đã validate vào topic traffic.validated"""
        return self._send(config.kafka.validated_topic, data, key)
    
    def send_features(self, data: Dict[str, Any], key: Optional[str] = None):
        """Gửi features đã extract vào topic traffic.features"""
        return self._send(config.kafka.features_topic, data, key)
    
    def _send(self, topic: str, data: Dict[str, Any], key: Optional[str] = None):
        """
        Gửi message vào Kafka topic
        
        Args:
            topic: Tên topic
            data: Dữ liệu
            key: Message key
        """
        try:
            future = self.producer.send(topic, value=data, key=key)
            
            # Block để đảm bảo message đã được gửi
            record_metadata = future.get(timeout=10)
            
            self.logger.debug(
                f"Sent to {topic} - "
                f"partition: {record_metadata.partition}, "
                f"offset: {record_metadata.offset}"
            )
            return record_metadata
            
        except KafkaError as e:
            self.logger.error(f"❌ Kafka error when sending to {topic}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Error when sending to {topic}: {e}")
            raise
    
    def flush(self):
        """Flush tất cả pending messages"""
        if self.producer:
            self.producer.flush()
            self.logger.debug("Flushed all pending messages")
    
    def close(self):
        """Đóng producer"""
        if self.producer:
            self.producer.close()
            self.logger.info("Closed Kafka producer")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()