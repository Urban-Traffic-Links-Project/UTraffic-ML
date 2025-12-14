# src/data_processing/mongo/traffic_mongo_loader.py

from pymongo import MongoClient
import pandas as pd


class TrafficMongoLoader:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="bktraffic"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def load_segments(self):
        """Load road segments (geometry + metadata)."""
        col = self.db["Segments"]
        data = list(col.find({}, {"_id": 0}))
        return pd.DataFrame(data)

    def load_segment_status(self, limit=None):
        """Load traffic status (speed, jamLevel, travelTime)."""
        col = self.db["SegmentStatus"]
        cursor = col.find({}, {"_id": 0})
        if limit:
            cursor = cursor.limit(limit)
        return pd.DataFrame(list(cursor))

    def load_segment_reports(self, limit=None):
        """Load camera/sensor reports."""
        col = self.db["SegmentReports"]
        cursor = col.find({}, {"_id": 0})
        if limit:
            cursor = cursor.limit(limit)
        return pd.DataFrame(list(cursor))

    def load_nodes(self):
        """Load road nodes."""
        col = self.db["Nodes"]
        data = list(col.find({}, {"_id": 0}))
        return pd.DataFrame(data)

    def load_way_osm(self, limit=None):
        """Load OSM way geometries."""
        col = self.db["WayOSM"]
        cursor = col.find({}, {"_id": 0})
        if limit:
            cursor = cursor.limit(limit)
        return pd.DataFrame(list(cursor))

    def load_node_osm(self):
        """Load OSM nodes."""
        col = self.db["NodeOSM"]
        data = list(col.find({}, {"_id": 0}))
        return pd.DataFrame(data)

    def load_weather(self):
        """Optional external features."""
        col = self.db["WeatherInfo"]
        data = list(col.find({}, {"_id": 0}))
        return pd.DataFrame(data)

    def load_path_histories(self, limit=None):
        """Load historical movement/paths (optional for correlation)."""
        col = self.db["PathHistories"]
        cursor = col.find({}, {"_id": 0})
        if limit:
            cursor = cursor.limit(limit)
        return pd.DataFrame(list(cursor))


if __name__ == "__main__":
    loader = TrafficMongoLoader()

    print("Segments:", loader.load_segments().head())
    print("SegmentStatus:", loader.load_segment_status(limit=5).head())
    print("SegmentReports:", loader.load_segment_reports(limit=5).head())
