import gzip
import json
import time
from typing import Dict, List, Optional
import os
import requests
from dotenv import load_dotenv
load_dotenv()

class TomTomTrafficDataCollector:
    """
    Thu thập dữ liệu giao thông thực tế từ TomTom Traffic Stats API
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tomtom.com/traffic/trafficstats"
        
    def create_area_analysis_job(self, geometry: dict, date_from: str, date_to: str,
                                 job_name: str="Thu Duc Traffic Analysis",
                                 time_sets: List[Dict] = None) -> Optional[str]:
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
            # Tạo time sets mặc định cho giờ cao điểm và bình thường
            time_sets = [
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
            
        request_body = {
            "jobName": job_name,
            "distanceUnit": "KILOMETERS",
            "network": {
                "name": "Thu Duc Network",
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
        
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=request_body,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("responseStatus") == "OK":
                    job_id = result.get("jobId")
                    print(f"✅ Job created successfully! Job ID: {job_id}")
                    print(f"   Messages: {result.get('messages')}")
                    return job_id
                else:
                    print(f"❌ Job creation failed: {result}")
                    return None
            elif response.status_code == 400:
                result = response.json()
                # Nếu job đã tồn tại thì lấy luôn jobId từ response
                if "already created" in str(result.get("messages")) and result.get("jobId"):
                    job_id = result["jobId"]
                    print(f"ℹ️ Job already exists. Using existing Job ID: {job_id}")
                    return job_id
                else:
                    print(f"❌ API Error: {response.status_code} - {result}")
                    return None
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:    
            print(f"❌ Exception when creating job: {e}")
            return None
    
    def check_job_status(self, job_id: str) -> dict:
        """
        Kiểm tra trạng thái job
        """
        url = f"{self.base_url}/status/1/{job_id}?key={self.api_key}"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Status check error: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ Exception when checking status: {e}")
            return {}
    
    def wait_for_job_completion(self, job_id: str, max_wait_minutes: int = 60) -> Optional[dict]:
        """
        Chờ job hoàn thành và trả về kết quả
        """
        print(f"⏳ Waiting for job {job_id} to complete...")
        
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_wait_seconds:
                print(f"⏰ Timeout after {max_wait_minutes} minutes")
                return None
            
            status = self.check_job_status(job_id)
            job_state = status.get("jobState", "UNKNOWN")
            
            print(f"   Status: {job_state} (elapsed: {int(elapsed)}s)")
            
            if job_state == "DONE":
                print("Job completed successfully!")
                return status
            elif job_state in ["ERROR", "REJECTED", "CANCELLED"]:
                print(f"Job failed with state: {job_state}")
                return None
            elif job_state == "NEED_CONFIRMATION":
                print("Job needs manual confirmation")
                sample_url = status.get("sampleDetailsUrl")
                if sample_url:
                    print(f"   Sample details: {sample_url}")
                return status
            
            # Chờ trước khi check lại
            time.sleep(30)
    
    def download_results(self, job_id: str):
        """
        Download kết quả JSON từ job đã hoàn thành
        """
        from urllib.parse import urlparse
        
        PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir = os.path.join(PROJECT_DIR, "data", "raw", "tomtom_stats")
        
        os.makedirs(output_dir, exist_ok=True)
        
        status = self.check_job_status(job_id)
        
        if status.get("jobState") != "DONE":
            print(f"⚠️  Job is not DONE yet: {status.get('jobState')}")
            return None
        
        urls = status.get("urls", [])
        if not urls:
            print("❌ No result URLs found")
            return None
        
        json_url = None
        for url in urls:
            filename = urlparse(url).path
            if filename.endswith(".json"):
                json_url = url
                break
            if filename.endswith(".geojson"):
                json_url = url
                break

        if not json_url:
            print("⚠️ No JSON/GeoJSON result")
            return None
    
        print(f"📥 Downloading results from: {json_url}")
        
        try:
            response = requests.get(json_url, timeout=120)
            if response.status_code == 200:
                # Decompress gzip
                content = response.content
                try:
                    content = gzip.decompress(response.content)
                except OSError:
                    pass
                result_data = json.loads(content.decode("utf-8"))
                
                # Save to file
                output_file = os.path.join(output_dir, f"job_{job_id}_results.json")
                with open(output_file, 'w', encoding="utf-8") as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Results saved to: {output_file}")
                return result_data
            else:
                print(f"❌ Download failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Exception during download: {e}")
            return None

if __name__ == "__main__":
    API_KEY = os.getenv("TOMTOM_TRAFFIC_STATS_API_KEY", None)

    collector = TomTomTrafficDataCollector(API_KEY)

    # Ví dụ: polygon nhỏ quanh Thủ Đức
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

    # Tạo job phân tích cho tháng 8/2024
    job_id = collector.create_area_analysis_job(
        geometry=geometry,
        date_from="2024-08-01",
        date_to="2024-08-31",
        job_name="Thu Duc Test Job"
    )

    if job_id:
        # Chờ job hoàn tất
        status = collector.wait_for_job_completion(job_id, max_wait_minutes=10)

        if status and status.get("jobState") == "DONE":
            # Tải kết quả về local
            results = collector.download_results(job_id)

            if results:
                print("✅ Có dữ liệu TomTom, in thử vài dòng:")
                print(json.dumps(results, indent=2, ensure_ascii=False)[:1000])