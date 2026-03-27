# src/data_processing/preprocessors/feature_extractor.py
"""
OUTPUT FEATURES (~20):
    Base (từ TomTom):
        average_speed, harmonic_average_speed, median_speed, std_speed,
        average_travel_time, travel_time_ratio, sample_size
    Derived (speed-based):
        relative_speed, speed_reduction_ratio, speed_limit_ratio, speed_range,
        speed_skewness, speed_cv
    Congestion:
        congestion_index, free_flow_speed, free_flow_travel_time
    Temporal encoding:
        time_set_sin, time_set_cos, is_peak, is_morning_peak, is_evening_peak
    Dynamic (within-day rolling):
        speed_ma_3, speed_ma_6, speed_ema, speed_volatility
    Road static:
        distance, frc, speed_limit
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional

from utils.config import config
from utils.logger import LoggerMixin


# ── Danh sách features output cuối cùng ─────────────────────────────────────
# Thứ tự này được giữ nhất quán để model index features đúng.
FINAL_FEATURE_COLS: List[str] = [
    # Base TomTom features
    "average_speed",
    "harmonic_average_speed",
    "median_speed",
    "std_speed",
    "average_travel_time",
    "travel_time_ratio",
    "sample_size",
    # Derived speed features
    "relative_speed",
    "speed_reduction_ratio",
    "speed_limit_ratio",
    "speed_range",
    "speed_skewness",
    "speed_cv",
    # Congestion features
    "congestion_index",
    "free_flow_speed",
    "free_flow_travel_time",
    # Temporal encoding (cyclic)
    "time_set_sin",
    "time_set_cos",
    "is_peak",
    "is_morning_peak",
    "is_evening_peak",
    # Within-day rolling features (valid vì 24 slots/ngày có thứ tự)
    "speed_ma_3",
    "speed_ma_6",
    "speed_ema",
    "speed_volatility",
    # Road static features
    "distance",
    "frc",
    "speed_limit",
]

# Features sẽ được normalize (exclude categorical, cyclic, binary)
NORMALIZE_COLS: List[str] = [
    "average_speed", "harmonic_average_speed", "median_speed", "std_speed",
    "average_travel_time", "travel_time_ratio",
    "relative_speed", "speed_reduction_ratio", "speed_range",
    "speed_skewness", "speed_cv",
    "congestion_index", "free_flow_speed", "free_flow_travel_time",
    "speed_ma_3", "speed_ma_6", "speed_ema", "speed_volatility",
]

# Features normalize riêng với MinMaxScaler
MINMAX_COLS: List[str] = [
    "distance", "speed_limit", "speed_limit_ratio", "sample_size",
]


class FeatureExtractor(LoggerMixin):
    """
    Trích xuất features từ dữ liệu TomTom đã clean.

    Chỉ giữ lại features có ý nghĩa vật lý cho T-GCN / DTC-STGCN.
    Loại bỏ các features gây noise hoặc không có ý nghĩa với dữ liệu Probe Data.
    """

    def __init__(self):
        self.time_windows = getattr(config.features, "time_windows", [3, 6])

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline trích xuất đầy đủ.

        Args:
            df : DataFrame đã được DataCleaner.clean().

        Returns:
            DataFrame với ~28 features (FINAL_FEATURE_COLS + id/metadata cols).
        """
        self.logger.info(f"Extracting features from {len(df)} rows...")

        df_out = df.copy()

        # 1. Base derived features (free_flow_speed, free_flow_travel_time)
        df_out = self._extract_base_features(df_out)

        # 2. Speed-based features
        df_out = self._extract_speed_features(df_out)

        # 3. Congestion features
        df_out = self._extract_congestion_features(df_out)

        # 4. Temporal encoding từ time_set
        df_out = self._extract_temporal_features(df_out)

        # 5. Within-day rolling features (theo thứ tự time_slot trong ngày)
        df_out = self._extract_rolling_features(df_out)

        # 6. Giữ lại chỉ columns cần thiết + metadata
        df_out = self._select_final_columns(df_out)

        n_features = len([c for c in df_out.columns if c in FINAL_FEATURE_COLS])
        self.logger.info(
            f"✅ Feature extraction complete. "
            f"Features: {n_features}/{len(FINAL_FEATURE_COLS)} | "
            f"Total cols: {len(df_out.columns)}"
        )
        return df_out

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _extract_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính free_flow_speed và free_flow_travel_time."""

        # free_flow_speed = 95th percentile per segment (phản ánh điều kiện thông thoáng)
        if "free_flow_speed" not in df.columns and "average_speed" in df.columns:
            if "segment_id" in df.columns:
                df["free_flow_speed"] = df.groupby("segment_id")["average_speed"].transform(
                    lambda x: x.quantile(0.95) if len(x) >= 4 else x.max()
                )
            else:
                df["free_flow_speed"] = df["average_speed"].quantile(0.95)

        # free_flow_travel_time = distance / free_flow_speed (đơn vị: giờ nếu distance km)
        if "free_flow_travel_time" not in df.columns:
            if "distance" in df.columns and "free_flow_speed" in df.columns:
                df["free_flow_travel_time"] = df["distance"] / (
                    df["free_flow_speed"].clip(lower=1.0)
                )
            elif "distance" in df.columns and "speed_limit" in df.columns:
                df["free_flow_travel_time"] = df["distance"] / (
                    df["speed_limit"].clip(lower=1.0)
                )

        return df

    def _extract_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trích xuất speed-based features.

        LOẠI BỎ so v1:
            - speed_derivative, speed_acceleration, speed_rate_of_change:
              Không có ý nghĩa vật lý (24 slots/ngày không liên tục theo thời gian thực).
            - speed_percentile: Chỉ hữu ích cho ranking, không cần trong GCN.
            - speed_zscore: Thay bằng speed_cv (coefficient of variation) nhất quán hơn.
        """
        if "average_speed" not in df.columns:
            return df

        # relative_speed = actual / free_flow (mức độ chậm so với điều kiện lý tưởng)
        if "free_flow_speed" in df.columns:
            df["relative_speed"] = df["average_speed"] / (
                df["free_flow_speed"].clip(lower=1.0)
            )
            df["speed_reduction_ratio"] = (1.0 - df["relative_speed"]).clip(lower=0.0)

        # speed_limit_ratio = actual / speed_limit (mức độ tuân thủ luật)
        if "speed_limit" in df.columns:
            df["speed_limit_ratio"] = df["average_speed"] / (
                df["speed_limit"].clip(lower=1.0)
            )

        # speed_range = average - harmonic (phản ánh phân tán vận tốc trong nhóm xe)
        if "harmonic_average_speed" in df.columns:
            df["speed_range"] = (df["average_speed"] - df["harmonic_average_speed"]).clip(
                lower=0.0
            )

        # speed_skewness = (mean - median) / std (phân phối vận tốc lệch phải/trái)
        if "median_speed" in df.columns and "std_speed" in df.columns:
            df["speed_skewness"] = (df["average_speed"] - df["median_speed"]) / (
                df["std_speed"].clip(lower=0.1)
            )

        # speed_cv = coefficient of variation = std / mean (normalized variability)
        if "std_speed" in df.columns:
            df["speed_cv"] = df["std_speed"] / df["average_speed"].clip(lower=1.0)

        return df

    def _extract_congestion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trích xuất congestion index."""

        # congestion_index = (actual_travel_time / free_flow_travel_time) - 1
        # = 0 khi không kẹt, > 0 khi kẹt (vd 0.5 = chậm 50%)
        if (
            "average_travel_time" in df.columns
            and "free_flow_travel_time" in df.columns
        ):
            df["congestion_index"] = (
                df["average_travel_time"] / df["free_flow_travel_time"].clip(lower=0.001) - 1.0
            ).clip(lower=0.0)

        # Đảm bảo travel_time_ratio có trong df (từ TomTom trực tiếp)
        # Nếu không có thì tính lại
        if "travel_time_ratio" not in df.columns:
            if "average_travel_time" in df.columns and "free_flow_travel_time" in df.columns:
                df["travel_time_ratio"] = df["average_travel_time"] / (
                    df["free_flow_travel_time"].clip(lower=0.001)
                )

        return df

    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trích xuất temporal features từ time_set.

        Cyclic encoding: time_set_sin, time_set_cos (giữ nguyên).
        Binary: is_peak, is_morning_peak, is_evening_peak.

        LOẠI BỎ so v1:
            - time_set_encoded: không cần khi đã có sin/cos.
            - hour, day_of_week: không có từ dữ liệu TomTom Stats (chỉ có slots).
        """
        if "time_set" not in df.columns:
            return df

        # Parse time slot → giờ trong ngày
        # Slot_HHMM → float giờ (vd "Slot_0715" → 7.25)
        def _slot_to_hour(ts) -> Optional[float]:
            s = str(ts)
            if "Slot_" in s:
                code = s.replace("Slot_", "").strip()
                if len(code) == 4:
                    try:
                        h = int(code[:2])
                        m = int(code[2:])
                        return h + m / 60.0
                    except ValueError:
                        pass
            return None

        hours = df["time_set"].map(_slot_to_hour)
        valid_mask = hours.notna()

        if valid_mask.any():
            # Cyclic encoding (24h cycle)
            df.loc[valid_mask, "time_set_sin"] = np.sin(
                2 * np.pi * hours[valid_mask] / 24.0
            )
            df.loc[valid_mask, "time_set_cos"] = np.cos(
                2 * np.pi * hours[valid_mask] / 24.0
            )

            # Peak hours (giờ cao điểm Quận 1, HCMC)
            df.loc[valid_mask, "is_morning_peak"] = (
                hours[valid_mask].between(7.0, 10.0)
            ).astype(np.float32)
            df.loc[valid_mask, "is_evening_peak"] = (
                hours[valid_mask].between(15.0, 18.0)
            ).astype(np.float32)
            df.loc[valid_mask, "is_peak"] = (
                df.loc[valid_mask, "is_morning_peak"]
                + df.loc[valid_mask, "is_evening_peak"]
            ).clip(upper=1.0)
        else:
            # Fallback: nếu không parse được, dùng index-based encoding
            unique_ts = df["time_set"].unique()
            ts_map = {ts: i for i, ts in enumerate(sorted(unique_ts))}
            ts_idx = df["time_set"].map(ts_map).fillna(0)
            n = max(len(unique_ts), 1)
            df["time_set_sin"] = np.sin(2 * np.pi * ts_idx / n)
            df["time_set_cos"] = np.cos(2 * np.pi * ts_idx / n)
            df["is_morning_peak"] = 0.0
            df["is_evening_peak"] = 0.0
            df["is_peak"] = 0.0

        return df

    def _extract_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling features TRONG TỪNG NGÀY per segment.

        Chỉ tính rolling khi có >= 2 time slots cho cùng segment + date.
        Đây là rolling window hợp lý vì 24 slots trong 1 ngày có thứ tự thời gian thực.

        LOẠI BỎ so v1:
            - speed_ma_12: window 12 slots = 3 giờ, quá lớn cho pattern cao điểm.
            - speed_derivative, speed_acceleration: không có ý nghĩa vật lý.
        """
        if "average_speed" not in df.columns or "segment_id" not in df.columns:
            return df

        # Sort theo segment + date + time_slot để rolling đúng thứ tự
        sort_cols = ["segment_id"]
        if "date_from" in df.columns:
            sort_cols.append("date_from")
        if "time_set" in df.columns:
            sort_cols.append("time_set")

        df = df.sort_values(sort_cols).reset_index(drop=True)

        # Group key: segment + date (rolling chỉ trong 1 ngày)
        if "date_from" in df.columns:
            group_key = ["segment_id", "date_from"]
        else:
            group_key = ["segment_id"]

        def _rolling_group(grp: pd.Series, window: int) -> pd.Series:
            return grp.rolling(window, min_periods=1).mean()

        def _ema_group(grp: pd.Series) -> pd.Series:
            return grp.ewm(span=6, adjust=False).mean()

        def _volatility_group(grp: pd.Series, window: int) -> pd.Series:
            return grp.rolling(window, min_periods=2).std().fillna(0.0)

        speed_col = df["average_speed"]

        # MA_3: 3 slots × 15 phút = 45 phút window
        df["speed_ma_3"] = (
            df.groupby(group_key, sort=False)["average_speed"]
            .transform(lambda x: _rolling_group(x, 3))
        )

        # MA_6: 6 slots × 15 phút = 1.5 giờ window
        df["speed_ma_6"] = (
            df.groupby(group_key, sort=False)["average_speed"]
            .transform(lambda x: _rolling_group(x, 6))
        )

        # EMA (span=6 ≈ 1.5 giờ, nhạy với thay đổi gần hơn MA_6)
        df["speed_ema"] = (
            df.groupby(group_key, sort=False)["average_speed"]
            .transform(lambda x: _ema_group(x))
        )

        # Volatility: rolling std_6 (mức độ dao động tốc độ trong 1.5 giờ)
        df["speed_volatility"] = (
            df.groupby(group_key, sort=False)["average_speed"]
            .transform(lambda x: _volatility_group(x, 6))
        )

        return df

    def _select_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chỉ giữ lại FINAL_FEATURE_COLS + các metadata columns cần thiết.

        Metadata columns: segment_id, new_segment_id, street_name,
                          time_set, date_from, latitude/longitude.
        """
        metadata_cols = [
            "segment_id", "new_segment_id", "street_name",
            "time_set", "date_from", "date_range",
            "latitude", "longitude",
            "raw_latitude", "raw_longitude",
        ]

        keep_cols = []
        for col in metadata_cols:
            if col in df.columns:
                keep_cols.append(col)

        missing_features = []
        for col in FINAL_FEATURE_COLS:
            if col in df.columns:
                keep_cols.append(col)
            else:
                missing_features.append(col)

        if missing_features:
            self.logger.debug(
                f"Features không có trong DataFrame (sẽ skip): {missing_features}"
            )

        return df[keep_cols]


# =========================================================================
# MODULE-LEVEL CONSTANTS (export cho các module khác dùng)
# =========================================================================

__all__ = [
    "FeatureExtractor",
    "FINAL_FEATURE_COLS",
    "NORMALIZE_COLS",
    "MINMAX_COLS",
]