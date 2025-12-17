# src/data_processing/streaming/kafka_consumer.py
import json
from typing import Callable, Optional, List
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from ...utils.config import config
from ...utils.logger import LoggerMixin

class TrafficDataConsumer(LoggerMixin):
    """
    Kafka Consumer để nhận và xử lý dữ liệu giao thông từ Kafka topics
    """
    
    def __init__(
        self,
        topics: List[str],
        group_id: Optional[str] = None,
        bootstrap_servers: Optional[str] = None,
        auto_offset_reset: Optional[str] = None
    ):
        self.topics = topics
        self.group_id = group_id or config.kafka.consumer_group
        self.bootstrap_servers = bootstrap_servers or config.kafka.bootstrap_servers
        self.auto_offset_reset = auto_offset_reset or config.kafka.auto_offset_reset
        self.consumer = None
        self._connect()
    
    def _connect(self):
        """Khởi tạo Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=config.kafka.enable_auto_commit,
                max_poll_records=config.kafka.max_poll_records,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None
            )
            self.logger.info(
                f"✅ Connected to Kafka topics {self.topics} "
                f"with group {self.group_id}"
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Kafka: {e}")
            raise
    
    def consume(self, processor: Callable, max_messages: Optional[int] = None):
        """
        Consume messages và xử lý với processor function
        
        Args:
            processor: Function nhận (key, value, topic, partition, offset) 
                      và trả về True nếu xử lý thành công
            max_messages: Số message tối đa cần consume (None = vô hạn)
        """
        processed_count = 0
        
        try:
            for message in self.consumer:
                try:
                    success = processor(
                        key=message.key,
                        value=message.value,
                        topic=message.topic,
                        partition=message.partition,
                        offset=message.offset
                    )
                    
                    if success:
                        # Manual commit nếu xử lý thành công
                        self.consumer.commit()
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            self.logger.info(f"Processed {processed_count} messages")
                        
                        if max_messages and processed_count >= max_messages:
                            self.logger.info(f"Reached max_messages limit: {max_messages}")
                            break
                    
                except Exception as e:
                    self.logger.error(
                        f"❌ Error processing message at "
                        f"{message.topic}:{message.partition}:{message.offset} - {e}"
                    )
                    # Không commit khi có lỗi, sẽ retry message này
                    continue
                    
        except KeyboardInterrupt:
            self.logger.info("⏹️ Consumer interrupted by user")
        except KafkaError as e:
            self.logger.error(f"❌ Kafka error: {e}")
            raise
        finally:
            self.logger.info(f"Total processed: {processed_count} messages")
    
    def close(self):
        """Đóng consumer"""
        if self.consumer:
            self.consumer.close()
            self.logger.info("Closed Kafka consumer")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RawDataConsumer(TrafficDataConsumer):
    """Consumer chuyên dụng cho traffic.raw topic"""
    
    def __init__(self):
        super().__init__(topics=[config.kafka.raw_topic])


class ValidatedDataConsumer(TrafficDataConsumer):
    """Consumer chuyên dụng cho traffic.validated topic"""
    
    def __init__(self):
        super().__init__(topics=[config.kafka.validated_topic])


class FeaturesConsumer(TrafficDataConsumer):
    """Consumer chuyên dụng cho traffic.features topic"""
    
    def __init__(self):
        super().__init__(topics=[config.kafka.features_topic])