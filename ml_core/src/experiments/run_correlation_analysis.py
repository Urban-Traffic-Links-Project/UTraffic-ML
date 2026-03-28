"""
Run T-GCN Correlation Analysis  [FIXED VERSION v2]
===================================================
Script entry-point để:
  1. Load T-GCN checkpoint đã train
  2. Load model-ready data (NPZ)
  3. Xây adj thực sự từ tương quan tốc độ
  4. Extract speed predictions (mode="speed_per_step") — không dùng hidden states
  5. Tính ma trận tương quan với "detrended_pearson" để loại bỏ common trend
  6. Save + vẽ heatmap

Cách chạy:
    cd ml_core/src
    python experiments/run_correlation_analysis.py

============================================================
THAY ĐỔI SO VỚI PHIÊN BẢN TRƯỚC:
============================================================
1. mode đổi thành "speed_per_step":
        Mỗi (batch × pred_step) = 1 sample → W = 30 × 12 = 360 windows
        (thay vì W=30 với mode="hidden").
        Nhiều windows hơn → correlation đáng tin hơn.

2. CORR_METHOD đổi thành "detrended_pearson":
        Trừ global mean (common trend của tất cả nút) trước khi tính Pearson.
        Loại bỏ artifact "tất cả nút cùng tăng/giảm theo giờ".

3. FIX double-normalize:
        Model.forward() nhận adj RAW — tự normalize nội bộ.
        extract_hidden_states() không truyền adj_norm nữa.

4. Giải thích kỳ vọng kết quả mới:
        mean ≈ 0.1 - 0.5  (thay vì 0.96 — corr thực tế sau detrend)
        std  ≈ 0.2 - 0.4  (có sự phân hoá rõ ràng giữa các nút)
        min  có thể âm    (nút có traffic ngược pha)
============================================================
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.T_GCN.tgcn  import TGCN
from utils.data_loader   import DataManager
from analysis.correlation_extractor import (
    TGCNCorrelationExtractor,
    build_speed_correlation_adj,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_correlation")

# ==============================================================================
# CONFIG
# ==============================================================================

CHECKPOINT_PATH = _ROOT / "checkpoints" / "T-GCN" / "best_model.pth"

# FIX: Dùng "speed_per_step" thay vì "hidden"
# "speed_per_step": W = B × pred_len × num_loaders = 30 × 12 = 360 windows
# "hidden": W = 30 windows — quá ít, dễ bị giả tương quan
EXTRACTION_MODE = "speed_per_step"

# FIX: Dùng "detrended_pearson" thay vì "pearson_per_dim"
# "detrended_pearson": loại bỏ common trend toàn cục → corr thực tế
# "pearson_per_dim": vẫn bị ảnh hưởng bởi common trend nếu W nhỏ
CORR_METHOD = "detrended_pearson"

BATCH_SIZE  = 32
MAX_BATCHES = None
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_SEQ_LEN  = 12
DEFAULT_PRED_LEN = 12

# Adj params
ADJ_SPEED_FEATURE_IDX = 0
ADJ_THRESHOLD         = 0.3
ADJ_TOP_K             = None

# ==============================================================================


def _infer_model_dims(state_dict: dict) -> dict:
    dims = {}
    gc1 = state_dict.get("tgcn_cell.gcn.gc1.weight")
    if gc1 is not None:
        dims["input_dim"]      = int(gc1.shape[0])
        dims["gcn_hidden_dim"] = int(gc1.shape[1])
    gc2 = state_dict.get("tgcn_cell.gcn.gc2.weight")
    if gc2 is not None:
        dims["hidden_dim"] = int(gc2.shape[1])
    fc = state_dict.get("fc.weight")
    if fc is not None:
        dims["output_dim"] = int(fc.shape[0])
    dims["has_dec_proj"] = "dec_proj.weight" in state_dict
    return dims


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    pred_len: int = DEFAULT_PRED_LEN,
) -> tuple[TGCN, np.ndarray, dict]:
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    config     = ckpt.get("config", {})
    adj_raw    = ckpt["adj"]
    adj        = adj_raw if isinstance(adj_raw, np.ndarray) else adj_raw.numpy()
    adj        = adj.astype(np.float32)
    state_dict = ckpt["model_state_dict"]

    inferred = _infer_model_dims(state_dict)
    logger.info(f"Dims inferred from state_dict: {inferred}")

    N = adj.shape[0]
    if np.allclose(adj, np.eye(N), atol=1e-6):
        logger.warning(
            "⚠️  Checkpoint adj là ma trận đơn vị! "
            "Sẽ xây lại adj từ tương quan tốc độ trong dữ liệu."
        )

    def _get(key, default):
        return inferred.get(key, config.get(key, default))

    input_dim      = _get("input_dim",      1)
    hidden_dim     = _get("hidden_dim",     64)
    output_dim     = _get("output_dim",     1)
    gcn_hidden_dim = _get("gcn_hidden_dim", 64)
    num_nodes      = adj.shape[0]
    seq_len        = int(config.get("seq_len",  seq_len))
    pred_len       = int(config.get("pred_len", pred_len))

    logger.info(
        f"Reconstructing TGCN — nodes={num_nodes}, input_dim={input_dim}, "
        f"hidden_dim={hidden_dim}, output_dim={output_dim}, "
        f"gcn_hidden={gcn_hidden_dim}, seq={seq_len}, pred={pred_len}"
    )

    model = TGCN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        gcn_hidden_dim=gcn_hidden_dim,
        dropout=config.get("dropout", 0.0),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    merged_config = {
        **config,
        "input_dim": input_dim, "hidden_dim": hidden_dim,
        "output_dim": output_dim, "gcn_hidden_dim": gcn_hidden_dim,
        "num_nodes": num_nodes, "seq_len": seq_len, "pred_len": pred_len,
    }
    logger.info("Model loaded successfully.")
    return model, adj, merged_config


def main():
    logger.info("=" * 60)
    logger.info("T-GCN Correlation Analysis  [FIXED VERSION v2]")
    logger.info("=" * 60)

    if not CHECKPOINT_PATH.exists():
        logger.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
        return

    # ── 1. Load model ──────────────────────────────────────────────────────────
    model, adj_ckpt, config = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    # ── 2. Load data ───────────────────────────────────────────────────────────
    logger.info("Loading data …")
    dm = DataManager()
    dm.load_all()
    train_loader, val_loader, test_loader, _ = dm.prepare_for_training(
        batch_size=BATCH_SIZE,
        normalize=False,
    )
    logger.info(
        f"Loaders — train: {len(train_loader)}, "
        f"val: {len(val_loader)}, test: {len(test_loader)} batches"
    )

    # ── 3. Xây adj từ tương quan tốc độ ───────────────────────────────────────
    X_all_list = []
    for x, _ in train_loader:
        X_all_list.append(x.numpy())
    X_all = np.concatenate(X_all_list, axis=0)  # (W, seq, N, F)

    logger.info(f"Building speed-correlation adj from X_train: {X_all.shape}")
    adj_real = build_speed_correlation_adj(
        X_all,
        speed_feature_idx=ADJ_SPEED_FEATURE_IDX,
        threshold=ADJ_THRESHOLD,
        top_k=ADJ_TOP_K,
    )

    # ── 4. Build extractor với mode="speed_per_step" ──────────────────────────
    # FIX: mode="speed_per_step" thay vì "hidden"
    extractor = TGCNCorrelationExtractor(
        model=model,
        adj=adj_real,
        device=DEVICE,
        mode=EXTRACTION_MODE,          # "speed_per_step"
        results_dir=_ROOT / "data" / "results",
    )

    # ── 5. Extract + tính correlation với detrended_pearson ───────────────────
    logger.info(
        f"Extracting representations (mode={EXTRACTION_MODE}) "
        f"from train + val + test loaders …"
    )
    logger.info(
        f"  Kỳ vọng W = {(len(train_loader) + len(val_loader) + len(test_loader))} batches "
        f"× {BATCH_SIZE} batch_size × 12 pred_steps (với speed_per_step)"
    )
    corr = extractor.build_correlation_matrix(
        loaders=[train_loader, val_loader, test_loader],
        method=CORR_METHOD,            # "detrended_pearson"
        max_batches=MAX_BATCHES,
        detrend=True,
    )
    logger.info(f"Correlation matrix shape: {corr.shape}")

    # ── 6. Save ────────────────────────────────────────────────────────────────
    paths = extractor.save(tag="tgcn_v2")
    logger.info(f"NPZ → {paths['npz']}")
    logger.info(f"CSV → {paths['csv']}")

    # ── 7. Plot ────────────────────────────────────────────────────────────────
    png = extractor.plot(tag="tgcn_v2")
    if png:
        logger.info(f"PNG → {png}")

    # ── 8. Stats ───────────────────────────────────────────────────────────────
    N = corr.shape[0]
    off_diag = corr[~np.eye(N, dtype=bool)]
    logger.info("=" * 60)
    logger.info("Correlation stats (off-diagonal) — FIXED v2:")
    logger.info(f"  mean : {off_diag.mean():.4f}  (kỳ vọng: 0.1–0.5 sau detrend)")
    logger.info(f"  std  : {off_diag.std():.4f}   (kỳ vọng: 0.15–0.40 → có phân hoá rõ)")
    logger.info(f"  max  : {off_diag.max():.4f}")
    logger.info(f"  min  : {off_diag.min():.4f}   (kỳ vọng: có giá trị âm → traffic ngược pha)")
    logger.info("=" * 60)

    # Thêm: so sánh phân bố với version cũ
    n_high  = int((off_diag > 0.8).sum())
    n_mid   = int(((off_diag > 0.3) & (off_diag <= 0.8)).sum())
    n_low   = int((off_diag <= 0.3).sum())
    n_neg   = int((off_diag < 0).sum())
    n_total = len(off_diag)

    logger.info("Phân bố correlation (so sánh sau detrend):")
    logger.info(f"  > 0.8  (rất cao): {n_high:6d} / {n_total} = {n_high/n_total:.1%}")
    logger.info(f"  0.3–0.8 (trung bình): {n_mid:6d} / {n_total} = {n_mid/n_total:.1%}")
    logger.info(f"  ≤ 0.3  (thấp): {n_low:6d} / {n_total} = {n_low/n_total:.1%}")
    logger.info(f"  < 0    (âm):   {n_neg:6d} / {n_total} = {n_neg/n_total:.1%}")
    logger.info("=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()