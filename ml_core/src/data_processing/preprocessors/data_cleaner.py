# src/data_processing/preprocessors/data_cleaner.py
import pandas as pd
import numpy as np

from utils.config import config
from utils.logger import LoggerMixin


class DataCleaner(LoggerMixin):
    """
    Làm sạch dữ liệu giao thông: xử lý missing values và outliers.

    Pipeline thứ tự:
        0. Convert distance m → km
        1. Remove duplicates
        2. Remove invalid rows  ← lên trước để median/IQR không bị kéo lệch
        3. Handle missing values (interpolate per-segment → segment median → global median)
        4. Handle outliers (IQR per-segment với fallback global khi segment quá ít data)
    """

    # Ngưỡng IQR tối thiểu (km/h) để tránh clip quá mạnh trên segment ngắn
    _MIN_IQR = 1.0

    # Giá trị đánh dấu sample_size không biết (thay vì dùng median có thể gây xóa nhầm)
    _UNKNOWN_SAMPLE_SIZE = 1

    def __init__(self):
        self.z_threshold = config.data.z_score_threshold        # giữ để tương thích config
        self.iqr_multiplier = config.data.iqr_multiplier        # khuyến nghị: 2.5–3.0
        self.interpolation_gap_max = config.data.interpolation_gap_max

    # =========================================================================
    # PUBLIC
    # =========================================================================

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline làm sạch toàn bộ.

        Args:
            df: DataFrame cần clean.

        Returns:
            DataFrame đã được clean.
        """
        self.logger.info(f"Cleaning data: {len(df)} rows")
        original_len = len(df)

        df_clean = df.copy()

        # 0. Convert distance m → km
        df_clean = self._convert_distance(df_clean)

        # 1. Remove duplicates
        df_clean = self._remove_duplicates(df_clean)

        # 2. Remove invalid rows TRƯỚC — tránh dữ liệu rác ảnh hưởng median/IQR
        df_clean = self._remove_invalid_rows(df_clean)

        # 3. Handle missing values
        df_clean = self._handle_missing_values(df_clean)

        # 4. Detect and handle outliers
        df_clean = self._handle_outliers(df_clean)

        self.logger.info(
            f"Cleaning complete: {len(df_clean)} rows "
            f"({original_len - len(df_clean)} removed)"
        )
        return df_clean

    # =========================================================================
    # STEP 0 — CONVERT DISTANCE
    # =========================================================================

    def _convert_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'distance' not in df.columns:
            return df
        if df['distance'].notna().any():
            median_distance = df['distance'].median()
            if median_distance > 50:
                df['distance'] = df['distance'] / 1000.0
                self.logger.info("Converted distance from meters to kilometers")
        return df

    # =========================================================================
    # STEP 1 — REMOVE DUPLICATES
    # =========================================================================

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        id_cols = [c for c in ['segment_id', 'time_set', 'date_range'] if c in df.columns]
        if id_cols:
            df = df.drop_duplicates(subset=id_cols, keep='first')
        removed = before - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed} duplicate rows")
        return df

    # =========================================================================
    # STEP 2 — REMOVE INVALID ROWS
    # =========================================================================

    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xóa các dòng có dữ liệu không hợp lệ TRƯỚC khi tính thống kê.

        Lưu ý về sample_size:
            - Chỉ xóa khi có giá trị thực < min_sample_size.
            - NaN được giữ lại để xử lý ở bước handle_missing_values.
        """
        before = len(df)

        # Speed <= 0 là lỗi cảm biến / dữ liệu rác
        speed_cols = [c for c in df.columns if 'speed' in c.lower()]
        for col in speed_cols:
            invalid_mask = df[col].notna() & (df[col] <= 0)
            df = df[~invalid_mask]

        # Sample size quá nhỏ: chỉ xóa khi có giá trị cụ thể, không xóa NaN
        if 'sample_size' in df.columns:
            min_ss = 10
            df = df[df['sample_size'].isna() | (df['sample_size'] >= min_ss)]

        removed = before - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed} invalid rows")
        return df

    # =========================================================================
    # STEP 3 — HANDLE MISSING VALUES
    # =========================================================================

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý missing values theo thứ tự:
            1. Sort theo thời gian để interpolation đúng chiều.
            2. Interpolate tuyến tính PER SEGMENT (giới hạn gap).
            3. Fill còn lại bằng median của segment đó.
            4. Fallback: global median.
        """
        self.logger.info("Handling missing values...")

        # Sort đúng chiều thời gian trước khi interpolate
        sort_cols = [c for c in ['segment_id', 'date_from', 'time_set'] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        target_cols = [
            c for c in df.columns
            if 'speed' in c.lower() or 'travel_time' in c.lower()
        ]

        for col in target_cols:
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue

            self.logger.info(f"  {col}: {missing_count} missing values")

            if 'segment_id' in df.columns:
                # 1. Interpolate per segment
                df[col] = df.groupby('segment_id')[col].transform(
                    lambda x: x.interpolate(
                        method='linear',
                        limit=self.interpolation_gap_max,
                        limit_direction='both'
                    )
                )
                # 2. Fill còn lại bằng median của segment
                df[col] = df.groupby('segment_id')[col].transform(
                    lambda x: x.fillna(x.median())
                )
            else:
                df[col] = df[col].interpolate(
                    method='linear',
                    limit=self.interpolation_gap_max,
                    limit_direction='both'
                )

            # 3. Fallback global median
            df[col] = df[col].fillna(df[col].median())

        # sample_size: dùng giá trị sentinel thay vì median
        # Lý do: median có thể < 10 → fill rồi lại bị xóa ở _remove_invalid_rows
        #        (bước đó đã chạy rồi, nhưng để tránh nhầm trong các lần gọi sau)
        if 'sample_size' in df.columns:
            n_missing = df['sample_size'].isnull().sum()
            if n_missing > 0:
                df['sample_size'] = df['sample_size'].fillna(self._UNKNOWN_SAMPLE_SIZE)
                self.logger.info(
                    f"  sample_size: {n_missing} missing → filled with sentinel={self._UNKNOWN_SAMPLE_SIZE}"
                )

        return df

    # =========================================================================
    # STEP 4 — HANDLE OUTLIERS
    # =========================================================================

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cap outliers bằng IQR per-segment.

        Fallback quan trọng:
            Nếu IQR của một segment < _MIN_IQR (segment quá ít data hoặc
            tốc độ quá đồng đều), dùng IQR global của cột đó thay thế.
            Tránh tình trạng clip toàn bộ giá trị hợp lệ trên segment ngắn.
        """
        self.logger.info("Handling outliers...")

        speed_cols = [c for c in df.columns if 'speed' in c.lower()]

        for col in speed_cols:
            # IQR global — dùng làm fallback cho segment có quá ít data
            global_Q1 = df[col].quantile(0.25)
            global_Q3 = df[col].quantile(0.75)
            global_IQR = global_Q3 - global_Q1

            if 'segment_id' in df.columns:
                seg_Q1 = df.groupby('segment_id')[col].transform(lambda x: x.quantile(0.25))
                seg_Q3 = df.groupby('segment_id')[col].transform(lambda x: x.quantile(0.75))
                seg_IQR = seg_Q3 - seg_Q1

                # Fallback: segment có IQR quá nhỏ → dùng global IQR
                use_global = seg_IQR < self._MIN_IQR
                effective_IQR = seg_IQR.where(~use_global, global_IQR)
                effective_Q1  = seg_Q1.where(~use_global, global_Q1)
                effective_Q3  = seg_Q3.where(~use_global, global_Q3)

                n_fallback = use_global.sum()
                if n_fallback > 0:
                    self.logger.info(
                        f"  {col}: {n_fallback} rows used global IQR fallback "
                        f"(segment IQR < {self._MIN_IQR} km/h)"
                    )
            else:
                effective_IQR = global_IQR
                effective_Q1  = global_Q1
                effective_Q3  = global_Q3

            lower_bound = effective_Q1 - self.iqr_multiplier * effective_IQR
            upper_bound = effective_Q3 + self.iqr_multiplier * effective_IQR

            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = int(outliers.sum())

            if outlier_count > 0:
                self.logger.info(f"  {col}: {outlier_count} outliers capped (per-segment IQR)")
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

        return df