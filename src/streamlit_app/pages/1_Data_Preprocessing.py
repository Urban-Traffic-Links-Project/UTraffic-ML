# pages/1_Data_Preprocessing.py
import streamlit as st
import pandas as pd
from pathlib import Path

from data_processing.storage.parquet_writer import ParquetReader
from src_ui.demo_data import make_demo_features
from src_ui.ui_helpers import summarize_df, missing_report

st.set_page_config(page_title="Data Preprocessing (Demo)", layout="wide")

# ---------- Loaders ----------
@st.cache_data(show_spinner=True)
def load_dataset_parquet(parquet_base: str) -> pd.DataFrame:
    """Đọc output thật từ Parquet (traffic_features). Không in st.* trong cache."""
    reader = ParquetReader(base_path=Path(parquet_base))
    df = reader.read_features()  # đọc traffic_features
    return df

@st.cache_data(show_spinner=True)
def load_dataset_demo() -> pd.DataFrame:
    """Dữ liệu giả để demo UI nếu chưa có parquet."""
    return make_demo_features()

# ---------- UI ----------
st.title("Data Preprocessing")
st.caption("Demo quy trình: Validation → Cleaning → Feature Extraction → Normalization")

with st.sidebar:
    st.header("Demo Controls")
    data_source = st.radio("Nguồn dữ liệu", ["parquet", "demo"], index=0)

    parquet_base = st.text_input(
        "Parquet base path",
        value=r"D:\Đồ án chuyên ngành HK251\Urban-Traffic-Links\data\processed\parquet",
        help="Thư mục chứa các table parquet: raw_traffic_data/, validated_traffic_data/, traffic_features/ ..."
    )

    st.divider()
    show_debug = st.checkbox("Hiển thị debug path", value=True)

# ---------- Load ----------
if data_source == "parquet":
    base_path = Path(parquet_base)

    if show_debug:
        st.write("📂 Parquet base path:")
        st.code(str(base_path))

        if base_path.exists():
            folders = [p.name for p in base_path.iterdir() if p.is_dir()]
            st.write("📂 Folders trong parquet_dir:")
            st.code(folders if folders else "❌ Không có folder nào")
        else:
            st.error("❌ parquet_base path không tồn tại")

    # Kiểm tra folder traffic_features có tồn tại không
    if (not base_path.exists()) or (not (base_path / "traffic_features").exists()):
        st.warning("Chưa tìm thấy `traffic_features/` trong parquet_dir. Chuyển sang DEMO để bạn vẫn xem được giao diện.")
        df = load_dataset_demo()
        data_source_effective = "demo"
    else:
        df = load_dataset_parquet(parquet_base)
        data_source_effective = "parquet"

else:
    df = load_dataset_demo()
    data_source_effective = "demo"

if df is None or df.empty:
    st.error("Dataset rỗng hoặc không đọc được. Hãy chạy pipeline để tạo traffic_features hoặc kiểm tra lại parquet_base.")
    st.stop()

# KPI nhanh (demo-friendly)
info = summarize_df(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nguồn dữ liệu", data_source_effective)
c2.metric("Rows", f"{info['rows']:,}")
c3.metric("Columns", f"{info['cols']:,}")
c4.metric("Segments", f"{info['n_segments']:,}" if info["n_segments"] is not None else "—")

# ---------- Tabs: demo đủ quy trình ----------
tab_overview, tab_validation, tab_cleaning, tab_features = st.tabs(
    ["1) Overview", "2) Validation", "3) Cleaning", "4) Features & Normalization"]
)

with tab_overview:
    st.subheader("Overview – Quy trình tiền xử lí")
    st.markdown(
        """
Trong demo này, hệ thống trình bày đầu ra sau tiền xử lí theo 4 bước:

1. **Validation**: kiểm tra tính hợp lệ (schema, range, dữ liệu thiếu…)
2. **Cleaning**: xử lý trùng lặp, missing, outliers, dòng không hợp lệ
3. **Feature Extraction**: tạo đặc trưng theo thời gian & giao thông (speed/congestion/dynamic…)
4. **Normalization**: chuẩn hoá đặc trưng số để phục vụ phân tích/mô hình

        """
    )

    st.subheader("Xem nhanh dữ liệu đầu ra")
    st.dataframe(df.head(30), use_container_width=True)

    # Nếu là parquet, hiển thị info đơn giản thay vì get_table_info (tránh crash do pyarrow version)
    if data_source_effective == "parquet":
        st.subheader("Parquet (quick check)")
        st.write("Đang đọc table: `traffic_features`")
        st.code(str(Path(parquet_base) / "traffic_features"))

with tab_validation:
    st.subheader("Validation – Kiểm tra chất lượng dữ liệu (demo)")
    st.markdown(
        """
Demo validation tập trung vào 2 phần dễ hiểu:
- **Missing values** theo từng cột
- Một vài **rule cơ bản** để phát hiện dữ liệu bất thường (ví dụ: speed <= 0)
        """
    )

    st.write("**Missing report**")
    rep = missing_report(df)
    if rep.empty:
        st.success("Không có missing values trong dataset hiện tại.")
    else:
        st.dataframe(rep, use_container_width=True)

    # Rule demo: average_speed > 0
    if "average_speed" in df.columns:
        invalid_speed = df[df["average_speed"] <= 0]
        st.write("**Rule demo: average_speed > 0**")
        st.write(f"Found: {len(invalid_speed)} rows")
        if len(invalid_speed) > 0:
            st.dataframe(invalid_speed.head(50), use_container_width=True)

with tab_cleaning:
    st.subheader("Cleaning – Làm sạch dữ liệu (mô tả + kiểm tra nhanh)")
    st.markdown(
        """
Trong pipeline, bước cleaning thường bao gồm:
- Xoá trùng lặp theo khóa định danh (segment_id, time_set, date_range)
- Missing: nội suy tuyến tính + median theo segment + median toàn cục
- Outliers: kết hợp Z-score & IQR và **capping**
- Loại bỏ dòng không hợp lệ (speed=0, sample_size thấp)

Demo này hiển thị **thống kê nhanh** trên dữ liệu đầu ra.
        """
    )

    st.write("**Quick stats (numeric)**")
    numeric_desc = df.describe(include="number").transpose()
    st.dataframe(numeric_desc.head(30), use_container_width=True)

with tab_features:
    st.subheader("Features & Normalization – Xem schema đặc trưng (demo)")
    st.markdown(
        """
Ở bước feature extraction, pipeline tạo thêm các nhóm đặc trưng (tuỳ dataset có đủ cột):
- Speed-based: speed_limit_ratio, relative_speed...
- Temporal: hour, day_of_week, peak flags...
- Congestion: congestion_index, travel_time_ratio...
- Dynamic/statistical: derivative, moving average, z-score...

Bên dưới là danh sách cột + vài dòng mẫu.
        """
    )

    st.write("**Columns**")
    st.code("\n".join(sorted(df.columns.tolist())))

    st.subheader("Sample rows")
    st.dataframe(df.sample(min(len(df), 100)), use_container_width=True)
