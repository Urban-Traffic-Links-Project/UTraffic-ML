# src/data_processing/collectors/tomtom_collector.py
import gzip
import json
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse
import requests

from utils.config import config
from utils.logger import LoggerMixin

class TomTomTrafficDataCollector(LoggerMixin):
    """
    Thu thập dữ liệu giao thông từ TomTom Traffic Stats API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.tomtom.api_key
        self.base_url = config.tomtom.base_url
        self.timeout = config.tomtom.timeout
        self.max_retries = config.tomtom.max_retries
        self.retry_delay = config.tomtom.retry_delay
        
        if not self.api_key:
            raise ValueError("TomTom API key is required")
    
    def create_area_analysis_job(
        self, 
        geometry: dict, 
        date_from: str, 
        date_to: str,
        job_name: str = "Traffic Analysis",
        time_sets: Optional[List[Dict]] = None
    ) -> Optional[str]:
        """
        Tạo job phân tích giao thông cho một khu vực
        
        Args:
            geometry: GeoJSON MultiPolygon/Polygon
            date_from: Ngày bắt đầu (YYYY-MM-DD)
            date_to: Ngày kết thúc (YYYY-MM-DD)
            job_name: Tên job
            time_sets: Các time sets để phân tích
            
        Returns:
            job_id nếu thành công, None nếu thất bại
        """
        
        if time_sets is None:
            time_sets = self._get_default_time_sets()
        
        request_body = {
            "jobName": job_name,
            "distanceUnit": "KILOMETERS",
            "network": {
                "name": f"{job_name} Network",
                "geometry": geometry,
                "timeZoneId": "Asia/Ho_Chi_Minh",
                "frcs": ["0", "1", "2", "3", "4", "5", "6", "7"],
                "probeSource": "ALL"
            },
            "dateRange": {
                "name": f"{date_from} to {date_to}",
                "from": date_from,
                "to": date_to,
            },
            "timeSets": time_sets
        }
        
        url = f"{self.base_url}/areaanalysis/1?key={self.api_key}"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=request_body,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("responseStatus") == "OK":
                        job_id = result.get("jobId")
                        self.logger.info(f"✅ Job created successfully! Job ID: {job_id}")
                        return job_id
                    else:
                        self.logger.error(f"Job creation failed: {result}")
                        return None
                        
                elif response.status_code == 400:
                    result = response.json()
                    if "already created" in str(result.get("messages")) and result.get("jobId"):
                        job_id = result["jobId"]
                        self.logger.info(f"ℹ️ Job already exists. Using existing Job ID: {job_id}")
                        return job_id
                    else:
                        self.logger.error(f"API Error 400: {result}")
                        return None
                else:
                    self.logger.error(f"API Error {response.status_code}: {response.text}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Exception on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None
    
    def check_job_status(self, job_id: str) -> dict:
        """Kiểm tra trạng thái job"""
        url = f"{self.base_url}/status/1/{job_id}?key={self.api_key}"
        
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Status check error: {response.status_code}")
                return {}
        except Exception as e:
            self.logger.error(f"Exception when checking status: {e}")
            return {}
    
    def wait_for_job_completion(
        self, 
        job_id: str, 
        max_wait_minutes: int = 60,
        check_interval: int = 30
    ) -> Optional[dict]:
        """
        Chờ job hoàn thành
        
        Args:
            job_id: ID của job
            max_wait_minutes: Thời gian chờ tối đa (phút)
            check_interval: Khoảng thời gian giữa các lần check (giây)
        """
        self.logger.info(f"⏳ Waiting for job {job_id} to complete...")
        
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        
        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_wait_seconds:
                self.logger.warning(f"⏰ Timeout after {max_wait_minutes} minutes")
                return None
            
            status = self.check_job_status(job_id)
            job_state = status.get("jobState", "UNKNOWN")
            
            self.logger.info(f"Status: {job_state} (elapsed: {int(elapsed)}s)")
            
            if job_state == "DONE":
                self.logger.info("✅ Job completed successfully!")
                return status
            elif job_state in ["ERROR", "REJECTED", "CANCELLED"]:
                self.logger.error(f"❌ Job failed with state: {job_state}")
                return None
            elif job_state == "NEED_CONFIRMATION":
                self.logger.warning("⚠️ Job needs manual confirmation")
                return status
            
            time.sleep(check_interval)
    
    def download_results(self, job_id: str, output_dir: Optional[str] = None) -> Optional[dict]:
        """
        Download kết quả JSON từ job đã hoàn thành
        
        Args:
            job_id: ID của job
            output_dir: Thư mục lưu kết quả (mặc định: config.data.raw_dir)
        """
        if output_dir is None:
            output_dir = config.data.raw_dir / "tomtom_stats"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        status = self.check_job_status(job_id)
        
        if status.get("jobState") != "DONE":
            self.logger.warning(f"⚠️ Job is not DONE yet: {status.get('jobState')}")
            return None
        
        urls = status.get("urls", [])
        if not urls:
            self.logger.error("❌ No result URLs found")
            return None
        
        # Tìm URL kết quả JSON/GeoJSON
        json_url = self._find_json_url(urls)
        if not json_url:
            self.logger.warning("⚠️ No JSON/GeoJSON result found")
            return None
        
        self.logger.info(f"📥 Downloading results from: {json_url}")
        
        try:
            response = requests.get(json_url, timeout=120)
            if response.status_code == 200:
                # Decompress if gzipped
                content = response.content
                try:
                    content = gzip.decompress(content)
                except OSError:
                    pass
                
                result_data = json.loads(content.decode("utf-8"))
                
                # Save to file
                output_file = output_dir / f"job_{job_id}_results.json"
                with open(output_file, 'w', encoding="utf-8") as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"💾 Results saved to: {output_file}")
                return result_data
            else:
                self.logger.error(f"❌ Download failed: {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"❌ Exception during download: {e}")
            return None
    
    def _get_default_time_sets(self) -> List[Dict]:
        """Tạo time sets mặc định cho giờ cao điểm và bình thường"""
        return [
            {
                "name": "Morning Peak",
                "timeGroups": [
                    {
                        "days": ["MON", "TUE", "WED", "THU", "FRI"],
                        "times": ["07:00-09:00"]
                    }
                ]
            },
            {
                "name": "Evening Peak",
                "timeGroups": [
                    {
                        "days": ["MON", "TUE", "WED", "THU", "FRI"],
                        "times": ["17:00-19:00"]
                    }
                ]
            },
            {
                "name": "Off Peak",
                "timeGroups": [
                    {
                        "days": ["MON", "TUE", "WED", "THU", "FRI"],
                        "times": ["10:00-16:00"]
                    }
                ]
            },
            {
                "name": "Weekend",
                "timeGroups": [
                    {
                        "days": ["SAT", "SUN"],
                        "times": ["00:00-24:00"]
                    }
                ]
            }
        ]
    
    def _find_json_url(self, urls: List[str]) -> Optional[str]:
        """Tìm URL của file JSON/GeoJSON trong danh sách URLs"""
        for url in urls:
            filename = urlparse(url).path
            if filename.endswith(".json"):
                return url
        return None