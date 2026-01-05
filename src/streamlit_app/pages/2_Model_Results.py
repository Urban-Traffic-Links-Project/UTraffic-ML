import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Model Results", layout="wide")

st.title("📈 Model Results")
st.caption("Hình minh hoạ kết quả mô hình: Confusion Matrix, PR/ROC Curve, Loss & Metrics")

# Đường dẫn thư mục ảnh (tương đối so với file page)
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "model_results"

# Danh sách ảnh (đổi tên file cho đúng với ảnh bạn đặt)
images = [
    ("Confusion Matrix (Test)", ASSET_DIR / "confusion_matrix.jpg"),
    ("PR Curve (Test)", ASSET_DIR / "pr_curve.jpg"),
    ("ROC Curve (Test)", ASSET_DIR / "roc_curve.jpg"),
    ("Loss Curves", ASSET_DIR / "loss_curves.jpg"),
    ("Metrics Curves (Accuracy/Precision/Recall/F1)", ASSET_DIR / "metrics_curves.jpg"),
]

# Sidebar: chọn cách hiển thị
with st.sidebar:
    st.header("Hiển thị")
    layout = st.radio("Layout", ["2 cột", "1 cột"], index=0)

def show_image(title: str, path: Path):
    st.subheader(title)
    if not path.exists():
        st.error(f"Không tìm thấy ảnh: {path}")
        return
    st.image(str(path), use_container_width=True)

if layout == "1 cột":
    for title, path in images:
        show_image(title, path)
else:
    # 2 cột: 2 + 2 + 1
    c1, c2 = st.columns(2)
    with c1:
        show_image(*images[0])
    with c2:
        show_image(*images[1])

    c3, c4 = st.columns(2)
    with c3:
        show_image(*images[2])
    with c4:
        show_image(*images[3])

    st.divider()
    show_image(*images[4])
