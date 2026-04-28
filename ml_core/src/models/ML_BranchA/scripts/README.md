# README — DMFM Full Matrix Output for Backend

Tài liệu này hướng dẫn cách dùng **2 file mới**:

```text
09_train_dmfm_export_model.py
10_precompute_dmfm_full_matrices_with_axis.py
```

Mục tiêu là tạo ra **ma trận tương quan dự đoán `R_pred`** từ DMFM, kèm theo **danh sách ID đoạn đường `segment_ids`** để backend biết mỗi ô trong ma trận là tương quan giữa đoạn đường nào với đoạn đường nào.

---

## 1. Đường dẫn đặt file

Đặt 2 file vào thư mục:

```text
ml_core/src/models/ML_BranchA/scripts/
```

Cấu trúc đúng:

```text
UTraffic-ML/
  ml_core/
    src/
      models/
        ML_BranchA/
          scripts/
            00_prepare_branchA_common_from_osm.py
            09_train_dmfm_export_model.py
            10_precompute_dmfm_full_matrices_with_axis.py
```

---

## 2. Luồng xử lý tổng quát

```text
00_prepare_branchA_common_from_osm.py
    ↓
Tạo R_series.npy từ dữ liệu OSM edge

09_train_dmfm_export_model.py
    ↓
Train DMFM model từ train/R_series.npy

10_precompute_dmfm_full_matrices_with_axis.py
    ↓
Sinh full ma trận tương quan dự đoán R_pred + segment_ids
```

---

# PHẦN A — DÀNH CHO NGƯỜI CHẠY MODEL

## 3. Bước 1: Chuẩn bị dữ liệu R_t

Chạy file prepare hiện có:

```powershell
python ml_core/src/models/ML_BranchA/scripts/00_prepare_branchA_common_from_osm.py --overwrite
```

Sau khi chạy xong, cần có folder:

```text
ml_core/src/models/ML_BranchA/data/05_branchA_prepare_segment_segment_rt/
```

Bên trong có:

```text
train/
  R_series.npy
  segment_ids.npy
  R_series_meta.csv

val/
  R_series.npy
  segment_ids.npy
  R_series_meta.csv

test/
  R_series.npy
  segment_ids.npy
  R_series_meta.csv
```

Trong đó:

```text
R_series.npy    = chuỗi ma trận tương quan R_t
segment_ids.npy = danh sách ID đoạn đường theo thứ tự hàng/cột của ma trận
```

---

## 4. Bước 2: Train DMFM model

Chạy:

```powershell
python ml_core/src/models/ML_BranchA/scripts/09_train_dmfm_export_model.py --train-samples 120 --rank 12 --overwrite
```

Trên Kaggle:

```python
%cd /kaggle/working/UTraffic-ML

!python -u ml_core/src/models/ML_BranchA/scripts/09_train_dmfm_export_model.py \
  --train-samples 120 \
  --rank 12 \
  --overwrite \
  2>&1 | tee logs_A09_train_dmfm.txt
```

### File 09 làm gì?

File này đọc:

```text
ml_core/src/models/ML_BranchA/data/05_branchA_prepare_segment_segment_rt/train/R_series.npy
```

Sau đó train DMFM và lưu model vào:

```text
ml_core/src/models/ML_BranchA/artifacts/dmfm_model/
```

Output chính:

```text
dmfm_model.npz
dmfm_config.json
segment_ids.npy
matrix_axis.csv
```

### `--train-samples 120` nghĩa là gì?

DMFM cần nhiều ma trận `R_t` để học quy luật thay đổi tương quan theo thời gian.

Tham số:

```text
--train-samples 120
```

nghĩa là lấy **120 ma trận R_t đại diện** từ tập train để fit DMFM.

Nếu muốn dùng toàn bộ train:

```powershell
--train-samples 0
```

Tuy nhiên, với số đoạn đường lớn, dùng toàn bộ có thể rất nặng RAM.

Khuyến nghị:

```text
Máy yếu / chạy thử:       --train-samples 80
Mức cân bằng:             --train-samples 120
Máy mạnh hơn:             --train-samples 200
Dùng toàn bộ train:       --train-samples 0
```

---

## 5. Bước 3: Sinh full ma trận tương quan dự đoán

Chạy:

```powershell
python ml_core/src/models/ML_BranchA/scripts/10_precompute_dmfm_full_matrices_with_axis.py --split test --horizons 1,3,6,9 --dtype float16 --save-npz-bundles --overwrite
```

Trên Kaggle:

```python
%cd /kaggle/working/UTraffic-ML

!python -u ml_core/src/models/ML_BranchA/scripts/10_precompute_dmfm_full_matrices_with_axis.py \
  --split test \
  --horizons 1,3,6,9 \
  --dtype float16 \
  --save-npz-bundles \
  --overwrite \
  2>&1 | tee logs_A10_precompute_dmfm_axis.txt
```

### File 10 làm gì?

File này đọc model đã train:

```text
ml_core/src/models/ML_BranchA/artifacts/dmfm_model/dmfm_model.npz
```

Sau đó đọc `R_series.npy` của split được chọn, ví dụ:

```text
ml_core/src/models/ML_BranchA/data/05_branchA_prepare_segment_segment_rt/test/R_series.npy
```

Rồi sinh:

```text
R_pred = DMFM(R_t, horizon)
```

với các horizon:

```text
1, 3, 6, 9
```

Nếu mỗi time slot là 15 phút:

```text
horizon 1 = dự đoán sau 15 phút
horizon 3 = dự đoán sau 45 phút
horizon 6 = dự đoán sau 90 phút
horizon 9 = dự đoán sau 135 phút
```

---

## 6. Output của file 10

Ví dụ với:

```text
split = test
horizon = 3
```

Output nằm ở:

```text
ml_core/src/models/ML_BranchA/artifacts/dmfm_predictions_full/test/h3/
```

Bên trong có:

```text
R_pred_series.npy
segment_ids.npy
matrix_axis.csv
R_pred_meta.csv
prediction_summary.json
bundles/
  dmfm_pred_test_h3_idx000000.npz
  dmfm_pred_test_h3_idx000001.npz
  ...
```

---

# PHẦN B — DÀNH CHO BACKEND

## 7. Quy ước quan trọng nhất

Backend chỉ cần nhớ quy ước sau:

```text
R_pred_series[pred_idx, i, j]
= hệ số tương quan dự đoán giữa segment_ids[i] và segment_ids[j]
```

Trong đó:

```text
pred_idx     = index của ma trận dự đoán theo thời gian
i            = index hàng
j            = index cột
segment_ids  = danh sách ID đoạn đường theo đúng thứ tự hàng/cột
```

Ví dụ:

```python
R_pred_series[pred_idx, 10, 25]
```

thì:

```python
source_id = segment_ids[10]
target_id = segment_ids[25]
corr      = R_pred_series[pred_idx, 10, 25]
```

Nghĩa là:

```text
corr là tương quan dự đoán giữa source_id và target_id
```

---

## 8. Cách backend đọc full matrix từ `.npy`

```python
import numpy as np

base_dir = "ml_core/src/models/ML_BranchA/artifacts/dmfm_predictions_full/test/h3"

R_series = np.load(f"{base_dir}/R_pred_series.npy", mmap_mode="r")
segment_ids = np.load(f"{base_dir}/segment_ids.npy")

pred_idx = 0
i = 10
j = 25

corr = float(R_series[pred_idx, i, j])
source_id = int(segment_ids[i])
target_id = int(segment_ids[j])

print("source_id:", source_id)
print("target_id:", target_id)
print("corr:", corr)
```

Kết quả có ý nghĩa:

```text
Ô [i, j] trong ma trận là tương quan giữa source_id và target_id.
```

---

## 9. Cách backend đọc từ file bundle `.npz`

Nếu khi chạy file 10 có bật:

```text
--save-npz-bundles
```

thì mỗi ma trận dự đoán sẽ có một file `.npz` riêng:

```text
bundles/dmfm_pred_test_h3_idx000000.npz
```

Backend đọc như sau:

```python
import numpy as np

data = np.load(
    "ml_core/src/models/ML_BranchA/artifacts/dmfm_predictions_full/test/h3/bundles/dmfm_pred_test_h3_idx000000.npz"
)

R_pred = data["R_pred"]
segment_ids = data["segment_ids"]

i = 10
j = 25

source_id = int(segment_ids[i])
target_id = int(segment_ids[j])
corr = float(R_pred[i, j])

print(source_id, target_id, corr)
```

Ưu điểm của bundle:

```text
Mỗi file .npz chứa đủ:
- R_pred
- segment_ids
- horizon
- timestamp_local
- pred_idx
- source_sample_id
```

Backend chỉ cần load một file là đủ thông tin.

---

## 10. Ý nghĩa các file metadata

### `segment_ids.npy`

Đây là file quan trọng nhất để map index ma trận sang ID đoạn đường.

```text
segment_ids[i] = ID đoạn đường ứng với hàng/cột i
```

### `matrix_axis.csv`

File này chỉ là bản CSV dễ đọc của `segment_ids.npy`.

Ví dụ:

```csv
matrix_index,segment_id,axis_role
0,1234,row_and_column
1,5678,row_and_column
2,9876,row_and_column
```

Nghĩa là:

```text
Hàng/cột 0 ứng với segment_id 1234
Hàng/cột 1 ứng với segment_id 5678
Hàng/cột 2 ứng với segment_id 9876
```

### `R_pred_meta.csv`

File này cho biết mỗi `pred_idx` ứng với timestamp nào.

Ví dụ:

```csv
pred_idx,source_sample_id,horizon,timestamp_local
0,0,3,2026-08-01 08:00:00
1,1,3,2026-08-01 08:15:00
```

Nghĩa là:

```text
R_pred_series[0] là ma trận dự đoán từ timestamp 2026-08-01 08:00:00 với horizon 3.
```

### `prediction_summary.json`

File mô tả nhanh output:

```text
split
horizon
dtype
shape
N
n_predictions
matrix_contract
```

---

## 11. Backend lấy top 30 đoạn đồng biến từ full matrix

Đồng biến nghĩa là hệ số tương quan dương:

```text
R[source, target] > 0
```

Không dùng trị tuyệt đối `abs(R)`, vì tương quan âm là nghịch biến.

Ví dụ code:

```python
import numpy as np

base_dir = "ml_core/src/models/ML_BranchA/artifacts/dmfm_predictions_full/test/h3"

R_series = np.load(f"{base_dir}/R_pred_series.npy", mmap_mode="r")
segment_ids = np.load(f"{base_dir}/segment_ids.npy")

pred_idx = 0
source_id = 1234
top_k = 30

node_id_to_index = {int(node_id): idx for idx, node_id in enumerate(segment_ids)}
source_idx = node_id_to_index[source_id]

row = np.asarray(R_series[pred_idx, source_idx], dtype=np.float32).copy()
row[source_idx] = -np.inf

valid = np.where(np.isfinite(row) & (row > 0))[0]
order = valid[np.argsort(row[valid])[::-1]][:top_k]

targets = []
for rank, target_idx in enumerate(order, start=1):
    if rank <= 10:
        color = "red"
    elif rank <= 20:
        color = "orange"
    else:
        color = "green"

    targets.append({
        "rank": rank,
        "target_id": int(segment_ids[target_idx]),
        "target_index": int(target_idx),
        "corr": float(row[target_idx]),
        "color": color,
    })

print(targets)
```

---

## 12. Backend show map như thế nào?

Output của DMFM cho backend biết:

```text
target_id
corr
color
```

Để vẽ lên map, backend/frontend cần join `target_id` với geometry OSM.

Nguồn geometry có thể là:

```text
matched_osm_edge_metadata.csv
```

hoặc file GeoJSON mà bạn export từ OSM graph.

Quy trình:

```text
target_id
  ↓
join với geometry theo model_node_id / segment_id
  ↓
vẽ LineString lên bản đồ
```

Màu gợi ý:

```text
source đoạn được click: màu đen
top 1–10: màu đỏ
top 11–20: màu cam/vàng
top 21–30: màu xanh
```

---

## 13. Khuyến nghị khi chạy full matrix

Với khoảng `N = 3697`:

```text
1 ma trận float16 ≈ 27 MB
1 ma trận float32 ≈ 55 MB
```

Nếu lưu nhiều timestamp và nhiều horizon sẽ rất nặng.

Khuyến nghị:

```text
- Dùng --dtype float16 cho demo/backend.
- Chỉ precompute những split/horizon cần demo.
- Nếu chỉ demo vài thời điểm, dùng --max-samples hoặc --sample-ids.
```

Ví dụ chỉ sinh 5 ma trận đầu tiên:

```powershell
python ml_core/src/models/ML_BranchA/scripts/10_precompute_dmfm_full_matrices_with_axis.py --split test --horizons 1,3,6,9 --max-samples 5 --dtype float16 --save-npz-bundles --overwrite
```

Ví dụ chỉ sinh một số sample cụ thể:

```powershell
python ml_core/src/models/ML_BranchA/scripts/10_precompute_dmfm_full_matrices_with_axis.py --split test --horizons 3 --sample-ids 0,10,20 --dtype float16 --save-npz-bundles --overwrite
```

---

## 14. Tóm tắt cho backend

Backend chỉ cần hiểu:

```text
1. R_pred_series.npy là chuỗi ma trận dự đoán.
2. segment_ids.npy là danh sách ID theo thứ tự hàng/cột.
3. R_pred_series[pred_idx, i, j] là tương quan giữa segment_ids[i] và segment_ids[j].
4. R_pred_meta.csv cho biết pred_idx ứng với timestamp nào.
5. matrix_axis.csv là file CSV để nhìn mapping index → segment_id.
```

Câu quan trọng nhất:

```text
Muốn biết ô [i,j] là tương quan giữa ID nào:
source_id = segment_ids[i]
target_id = segment_ids[j]
corr = R_pred[pred_idx, i, j]
```
