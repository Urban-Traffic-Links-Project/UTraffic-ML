# src_ui/demo_data.py
import pandas as pd
import numpy as np

def make_demo_features(n_segments: int = 20, n_rows: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    segment_ids = rng.integers(1000, 1000 + n_segments, size=n_rows)
    hours = rng.integers(0, 24, size=n_rows)
    dow = rng.integers(0, 7, size=n_rows)

    avg_speed = rng.normal(25, 7, size=n_rows).clip(1, 80)
    speed_limit = rng.choice([30, 40, 50, 60], size=n_rows)
    median_speed = (avg_speed + rng.normal(0, 2, size=n_rows)).clip(1, 90)
    std_speed = rng.normal(3, 1.2, size=n_rows).clip(0.1, 20)

    avg_tt = rng.normal(120, 30, size=n_rows).clip(1, 1000)     # seconds
    free_tt = (avg_tt / rng.uniform(1.0, 2.2, size=n_rows)).clip(1, 1000)

    sample_size = rng.integers(0, 120, size=n_rows)

    df = pd.DataFrame({
        "segment_id": segment_ids,
        "hour": hours,
        "day_of_week": dow,
        "average_speed": avg_speed,
        "median_speed": median_speed,
        "std_speed": std_speed,
        "speed_limit": speed_limit,
        "average_travel_time": avg_tt,
        "free_flow_travel_time": free_tt,
        "sample_size": sample_size,
        "latitude": rng.normal(10.775, 0.02, size=n_rows),
        "longitude": rng.normal(106.695, 0.03, size=n_rows),
    })

    # Một số feature “giống” feature_extractor của bạn
    df["speed_limit_ratio"] = df["average_speed"] / (df["speed_limit"] + 1e-6)
    df["congestion_index"] = df["average_travel_time"] / (df["free_flow_travel_time"] + 1e-6) - 1
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_morning_peak"] = ((df["hour"] >= 6) & (df["hour"] < 9)).astype(int)
    df["is_evening_peak"] = ((df["hour"] >= 16) & (df["hour"] < 19)).astype(int)
    df["is_peak"] = (df["is_morning_peak"] | df["is_evening_peak"]).astype(int)

    return df
