"""
T-GCN Correlation Extractor  [FIXED VERSION v2]
================================================
Fix toàn bộ các nguyên nhân gây ma trận tương quan giả (mean ≈ 0.96, min > 0.26).

============================================================
ROOT CAUSE ANALYSIS — tại sao corr mean=0.9655 (quá cao):
============================================================

PROBLEM 1 — Dữ liệu quá ít windows (W=30, nhưng thực chất chỉ 30 samples)
    Với 30 batches × batch_size=1, mỗi "window" là toàn bộ 385 nodes.
    30 temporal samples KHÔNG ĐỦ để phân biệt tương quan thực vs giả,
    đặc biệt khi data đã normalized → các nút có pattern tương tự nhau.

PROBLEM 2 — Hidden states sau LayerNorm có variance rất thấp qua thời gian
    TGCNCell áp dụng LayerNorm(h_new) sau mỗi bước.
    LayerNorm chuẩn hóa THEO CHIỀU hidden_dim cho mỗi (batch, node) riêng biệt,
    tức là chuẩn hóa spatial không phải temporal.
    Kết quả: các nút cùng "kiểu" traffic có activation vectors RẤT GIỐNG NHAU
    qua thời gian → Pearson per-dim ≈ 1.0.

PROBLEM 3 — pearson_per_dim tính corr(N, N) cho từng dim d
    series = H_t[d]  # shape (N, W) — nhưng W=30 rất nhỏ
    np.corrcoef(series) tính (N, N) correlation qua W=30 điểm.
    Với data đã normalize và LayerNorm outputs, chuỗi (W=30) gần như
    đơn điệu tăng/giảm → tất cả nút cùng chiều hướng → corr ≈ 1.

PROBLEM 4 — adj_norm được truyền vào extract_hidden_states()
    self.adj_norm đã normalize (D^{-1/2} A D^{-1/2}).
    Nhưng TGCN.forward() gọi self._get_adj_norm(adj) để normalize lại.
    Khi dùng extractor.model.tgcn_cell(x, h, self.adj_norm) TRỰC TIẾP,
    adj_norm đã normalize → đúng.
    Nhưng khi dùng self.model(x, self.adj_norm) thì bị DOUBLE-NORMALIZE.

PROBLEM 5 — Mismatch node count (305 vs 385)
    graph_structure N=305, model data N=385.
    Không align được → identity fallback adj → GCN = linear transform only.
    Mọi nút được xử lý HOÀN TOÀN ĐỘC LẬP bởi cùng weight matrix.
    Hidden states của tất cả nút đều là f(x_i, W) với cùng W → rất tương đồng.

============================================================
FIX SUMMARY (v2):
============================================================

FIX 1 — Dùng speed predictions thay vì hidden states để tính correlation
    Speed output (B, pred_len, N, 1) phản ánh trực tiếp dự báo tốc độ.
    Correlation trên speed predictions có ý nghĩa thực tế rõ ràng hơn
    hidden states sau LayerNorm.

FIX 2 — Dùng mean pooling qua pred_len trước khi tính correlation
    Rep = pred.mean(dim=1)  → (B, N, 1)  hoặc  (B, N, output_dim)
    Tránh artifact từ từng bước dự báo riêng lẻ.

FIX 3 — Tăng temporal diversity bằng cách flatten batch × time → windows
    Thay vì lấy 1 vector (N,) per batch, lấy pred_len vectors × B batches
    → tổng W = B × pred_len × num_loaders, đa dạng temporal hơn.

FIX 4 — Thêm mode "speed_per_step": dùng từng pred step làm 1 sample riêng
    H = (W × pred_len, N, output_dim) — tăng samples đáng kể.

FIX 5 — Sửa double-normalize: extract_hidden_states dùng adj RAW,
    không phải adj_norm, khi gọi self.model(x, adj).

FIX 6 — Thêm detrending option: trừ mean theo time trước khi tính Pearson
    Loại bỏ common trend (ví dụ tất cả nút đều tăng tốc độ theo giờ)
    → corr phản ánh pattern TƯƠNG ĐỐI, không phải xu hướng chung.

FIX 7 — Thêm method "partial_pearson" với detrending mạnh hơn
    Dùng residual sau khi trừ global mean signal (first PC) của tất cả nút.
    Tương đương với correlation sau khi loại bỏ common mode.
============================================================
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.T_GCN.tgcn import TGCN
from models.T_GCN.gcn  import normalize_adj

logger = logging.getLogger(__name__)


# ===========================================================================
# Utility: xây adj từ tương quan tốc độ (thay thế identity adj)
# ===========================================================================

def build_speed_correlation_adj(
    X: np.ndarray,
    speed_feature_idx: int = 0,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
) -> np.ndarray:
    """
    Xây dựng adjacency matrix dựa trên tương quan tốc độ giữa các nút.

    Args:
        X               : (total_windows, seq_len, num_nodes, num_features)
        speed_feature_idx: chỉ số feature tốc độ (0 = average_speed)
        threshold       : ngưỡng correlation để coi là cạnh (nếu top_k=None)
        top_k           : nếu đặt, mỗi nút chỉ giữ top_k láng giềng cao nhất

    Returns:
        adj: (N, N) float32 symmetric adjacency matrix
    """
    W, T, N, F = X.shape
    # Speed time-series cho mỗi nút: (W*T, N)
    speed = X[:, :, :, speed_feature_idx].reshape(W * T, N).astype(np.float64)

    # Pearson correlation matrix (N, N)
    corr = np.corrcoef(speed.T)   # (N, N)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)  # bỏ self-loop (normalize_adj thêm sau)

    if top_k is not None:
        adj = np.zeros_like(corr)
        for i in range(N):
            top_idx = np.argsort(corr[i])[-top_k:]
            adj[i, top_idx] = corr[i, top_idx]
        adj = np.maximum(adj, adj.T)
    else:
        adj = np.where(corr > threshold, corr, 0.0)
        adj = np.maximum(adj, adj.T)

    adj = adj.astype(np.float32)
    logger.info(
        f"Built speed-correlation adj: {N}×{N}, "
        f"edges={int((adj > 0).sum()) // 2}, "
        f"density={float((adj > 0).mean()):.3f}"
    )
    return adj


# ===========================================================================
# Main class (fixed v2)
# ===========================================================================

class TGCNCorrelationExtractor:
    """
    Trích xuất và tính ma trận tương quan nút×nút từ T-GCN đã train.

    THAY ĐỔI QUAN TRỌNG trong v2:
        1. Mode mặc định đổi thành "speed_per_step" — dùng speed predictions
           thay vì hidden states để tránh artifact từ LayerNorm.
        2. extract_hidden_states() truyền adj RAW vào model.forward()
           (không truyền adj_norm đã normalize để tránh double-normalize).
        3. Thêm detrend option cho compute_correlation() để loại bỏ
           common trend toàn cục trước khi tính Pearson.
        4. Thêm method "detrended_pearson" — khuyến nghị cho data đã normalize.

    Args:
        model      : TGCN instance (đã load weights).
        adj        : (N, N) adjacency matrix — raw binary/weighted, float32.
        device     : "cuda" / "cpu".
        mode       : "speed_per_step" (KHUYẾN NGHỊ) — dùng từng pred step riêng;
                     "speed"          — dùng mean qua pred_len;
                     "hidden"         — dùng hidden states (dễ bị giả tương quan).
        results_dir: Thư mục lưu kết quả.
    """

    def __init__(
        self,
        model: TGCN,
        adj: np.ndarray,
        device: str = "cpu",
        mode: Literal["hidden", "speed", "speed_per_step"] = "speed_per_step",
        results_dir: Optional[Path] = None,
    ):
        self.model  = model.eval().to(device)
        self.device = device
        self.mode   = mode

        # Kiểm tra adj
        N = adj.shape[0]
        is_identity = np.allclose(adj, np.eye(N), atol=1e-6)
        if is_identity:
            logger.warning(
                "⚠️  adj được truyền vào là ma trận đơn vị (identity)! "
                "GCN sẽ không có message passing. "
                "Hãy dùng build_speed_correlation_adj() để xây adj thực sự."
            )

        self.adj      = adj                          # FIX 5: lưu adj RAW
        self.adj_norm = normalize_adj(adj).to(device) # chỉ dùng khi cần trực tiếp

        if results_dir is None:
            results_dir = _SRC.parent / "data" / "results"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._hidden_matrix:      Optional[np.ndarray] = None
        self._corr_matrix:        Optional[np.ndarray] = None
        self._mean_hidden_matrix: Optional[np.ndarray] = None
        self.num_nodes = N

    # ------------------------------------------------------------------
    # Step 1 — Extract representations
    # ------------------------------------------------------------------

    def extract_hidden_states(
        self,
        loaders: Union[DataLoader, List[DataLoader]],
        max_batches: Optional[int] = None,
    ) -> np.ndarray:
        """
        Chạy model qua DataLoaders, thu thập representations.

        Mode "speed_per_step" (KHUYẾN NGHỊ):
            Mỗi (batch_item, pred_step) cho ra 1 sample → W = B × pred_len × num_loaders.
            Đây là cách tốt nhất để có đủ temporal diversity.

        Mode "speed":
            Mean qua pred_len → W = B × num_loaders.

        Mode "hidden":
            Encoder hidden state cuối → W = B × num_loaders.
            CẢNH BÁO: dễ bị giả tương quan cao do LayerNorm.

        FIX 5: Truyền adj RAW vào model.forward() — model tự normalize nội bộ.
               Tránh double-normalize khi dùng model(x, adj_norm).

        Returns:
            H: (W, num_nodes, dim) — dim=output_dim (speed) hoặc hidden_dim
        """
        if isinstance(loaders, DataLoader):
            loaders = [loaders]

        logger.info(
            f"Extracting {self.mode} representations from "
            f"{len(loaders)} loader(s) …"
        )

        all_reps: list[np.ndarray] = []
        self.model.eval()

        # FIX 5: Dùng adj RAW (tensor) để truyền vào model.forward()
        adj_raw_tensor = torch.FloatTensor(self.adj).to(self.device)

        with torch.no_grad():
            for loader_idx, loader in enumerate(loaders):
                batch_count = 0
                for batch_idx, (x, _) in enumerate(
                    tqdm(loader, desc=f"Loader {loader_idx+1}/{len(loaders)}")
                ):
                    if max_batches is not None and batch_count >= max_batches:
                        break

                    x = x.to(self.device)  # (B, seq, N, F)
                    batch_size, seq_len, num_nodes, _ = x.size()

                    if self.mode == "speed_per_step":
                        # FIX 1, 3, 4: Dùng từng pred step làm sample riêng biệt
                        # FIX 5: truyền adj_raw_tensor, không phải adj_norm
                        pred = self.model(x, adj_raw_tensor)  # (B, pred_len, N, out_dim)
                        # Reshape: (B × pred_len, N, out_dim)
                        B, pred_len, N, out_dim = pred.shape
                        rep = pred.reshape(B * pred_len, N, out_dim).cpu().numpy()
                        all_reps.append(rep)

                    elif self.mode == "speed":
                        # FIX 5: truyền adj_raw_tensor
                        pred = self.model(x, adj_raw_tensor)  # (B, pred_len, N, out_dim)
                        # FIX 2: mean qua pred_len
                        rep = pred.mean(dim=1).cpu().numpy()   # (B, N, out_dim)
                        all_reps.append(rep)

                    elif self.mode == "hidden":
                        # CẢNH BÁO: mode này dễ bị giả tương quan cao
                        # Dùng adj_norm (đã normalize) khi gọi tgcn_cell trực tiếp
                        h = torch.zeros(
                            batch_size, num_nodes, self.model.hidden_dim,
                            device=self.device
                        )
                        for t in range(seq_len):
                            h = self.model.tgcn_cell(x[:, t], h, self.adj_norm)
                        all_reps.append(h.cpu().numpy())  # (B, N, H)

                    else:
                        raise ValueError(f"Unknown mode: {self.mode}")

                    batch_count += 1

        H = np.concatenate(all_reps, axis=0)  # (W, N, dim)
        W = H.shape[0]
        logger.info(f"Extracted representation matrix: {H.shape}")
        logger.info(
            f"  Temporal samples (W): {W}  "
            f"(khuyến nghị W≥100 cho detrended_pearson, W≥30 cho per_dim)"
        )

        if W < 30:
            logger.warning(
                f"⚠️  Chỉ có W={W} temporal windows. "
                "Quá ít để tính correlation đáng tin. "
                "Hãy dùng mode='speed_per_step' để tăng số samples."
            )

        self._hidden_matrix      = H
        self._mean_hidden_matrix = H.mean(axis=0)
        return H

    # ------------------------------------------------------------------
    # Step 2 — Compute correlation  (FIX 6, 7)
    # ------------------------------------------------------------------

    def compute_correlation(
        self,
        H: Optional[np.ndarray] = None,
        method: Literal[
            "pearson",
            "pearson_per_dim",
            "detrended_pearson",
            "cosine",
        ] = "detrended_pearson",
        detrend: bool = True,
    ) -> np.ndarray:
        """
        Tính ma trận tương quan (N×N) giữa các nút.

        Methods:
            "detrended_pearson" (KHUYẾN NGHỊ):
                - Trừ global mean signal (common mode) khỏi mỗi chuỗi.
                - Tính Pearson trên residuals.
                - Loại bỏ artifact từ xu hướng chung toàn cục.

            "pearson_per_dim":
                - Với mỗi hidden dim d, tính Pearson trên chuỗi (W,).
                - Average qua D dims.
                - Tốt cho hidden states, NHƯNG vẫn có thể cao nếu W nhỏ.

            "pearson":
                - Flatten (W, D) → (W*D,) rồi corrcoef.
                - Giữ để backward compatibility.

            "cosine":
                - KHÔNG KHUYẾN NGHỊ với LayerNorm outputs.

        Args:
            H      : (W, N, D). Dùng cached nếu None.
            method : Method tính correlation.
            detrend: Nếu True, trừ mean qua W trước khi tính (loại bỏ common trend).
                     Áp dụng cho tất cả methods ngoài "detrended_pearson"
                     (vốn đã detrend mạnh hơn).

        Returns:
            corr: (N, N) float32, values in [-1, 1].
        """
        if H is None:
            if self._hidden_matrix is None:
                raise RuntimeError("Gọi extract_hidden_states() trước.")
            H = self._hidden_matrix

        W, N, D = H.shape
        logger.info(f"Computing {method} correlation for {N} nodes (W={W}, D={D}) …")

        if W < 10:
            logger.warning(
                f"⚠️  Chỉ có W={W} temporal windows. "
                "Khuyến nghị W≥100 để tránh correlation giả."
            )

        corr: np.ndarray

        # ── FIX 7: detrended_pearson ─────────────────────────────────────────
        if method == "detrended_pearson":
            # H: (W, N, D)
            # Mean pooling qua D → (W, N)
            S = H.mean(axis=2).astype(np.float64)  # (W, N)

            # FIX 6: Loại bỏ common trend (global mean qua tất cả nút)
            # global_trend: (W, 1) — mean của tất cả nút ở mỗi timestep
            global_trend = S.mean(axis=1, keepdims=True)  # (W, 1)

            # Residuals: (W, N)
            S_residual = S - global_trend  # trừ common mode

            # Tính Pearson trên residuals
            # corrcoef nhận (N, W) — transpose
            S_T = S_residual.T  # (N, W)

            # Thêm nhỏ noise để tránh constant series
            std_per_node = S_T.std(axis=1)
            nearly_const = std_per_node < 1e-8
            if nearly_const.any():
                n_const = int(nearly_const.sum())
                logger.warning(
                    f"{n_const} nút có variance ≈ 0 sau detrending. "
                    "Có thể là do quá ít temporal diversity. "
                    "Thêm tiny noise để tránh NaN."
                )
                rng = np.random.default_rng(42)
                S_T[nearly_const] += rng.normal(0, 1e-6, size=(n_const, W))

            corr = np.corrcoef(S_T).astype(np.float32)  # (N, N)
            logger.info(
                f"detrended_pearson: used mean-pooled speed residuals "
                f"(after removing global trend). W={W}, N={N}"
            )

        # ── pearson_per_dim ─────────────────────────────────────────────────
        elif method == "pearson_per_dim":
            # H: (W, N, D) → transpose → (D, N, W)
            H_t = H.astype(np.float64)

            # FIX 6: detrend option
            if detrend:
                # Trừ mean qua W cho mỗi (d, n)
                H_t = H_t - H_t.mean(axis=0, keepdims=True)

            H_t = H_t.transpose(2, 1, 0)  # (D, N, W)

            corr_sum = np.zeros((N, N), dtype=np.float64)
            valid_dims = 0

            for d in range(D):
                series = H_t[d]  # (N, W)
                stds = series.std(axis=1)
                if stds.min() < 1e-8:
                    continue
                c = np.corrcoef(series)  # (N, N)
                if not np.any(np.isnan(c)):
                    corr_sum += c
                    valid_dims += 1

            if valid_dims == 0:
                logger.warning("Tất cả hidden dims đều constant — fallback to detrended_pearson.")
                method = "detrended_pearson"
                return self.compute_correlation(H, method=method, detrend=detrend)

            corr = (corr_sum / valid_dims).astype(np.float32)
            logger.info(
                f"pearson_per_dim: averaged over {valid_dims}/{D} valid dims"
            )

        # ── pearson (flat) ──────────────────────────────────────────────────
        elif method == "pearson":
            # H: (W, N, D) → (N, W*D)
            X_flat = H.transpose(1, 0, 2).reshape(N, -1).astype(np.float64)
            if detrend:
                X_flat = X_flat - X_flat.mean(axis=1, keepdims=True)
            corr = np.corrcoef(X_flat).astype(np.float32)

        # ── cosine ──────────────────────────────────────────────────────────
        elif method == "cosine":
            logger.warning(
                "⚠️  Cosine similarity với hidden states sau LayerNorm "
                "sẽ luôn dương (mean ≈ 0.6-0.8). "
                "Khuyến nghị dùng 'detrended_pearson' thay thế."
            )
            X_flat = H.transpose(1, 0, 2).reshape(N, -1).astype(np.float64)
            norms = np.linalg.norm(X_flat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-8, norms)
            Xn = X_flat / norms
            corr = (Xn @ Xn.T).astype(np.float32)

        else:
            raise ValueError(f"Unknown method: {method}")

        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0).astype(np.float32)
        corr = np.nan_to_num(corr, nan=0.0)

        off_diag = corr[~np.eye(N, dtype=bool)]
        logger.info(
            f"Correlation matrix: {corr.shape}, "
            f"mean={off_diag.mean():.4f}, std={off_diag.std():.4f}, "
            f"max={off_diag.max():.4f}, min={off_diag.min():.4f}"
        )

        self._corr_matrix = corr
        return corr

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def build_correlation_matrix(
        self,
        loaders: Union[DataLoader, List[DataLoader]],
        method: Literal[
            "pearson",
            "pearson_per_dim",
            "detrended_pearson",
            "cosine",
        ] = "detrended_pearson",
        max_batches: Optional[int] = None,
        detrend: bool = True,
    ) -> np.ndarray:
        """
        Shortcut: extract_hidden_states() + compute_correlation().

        Mặc định dùng:
            mode = "speed_per_step" (set ở __init__)
            method = "detrended_pearson"
        """
        H = self.extract_hidden_states(loaders, max_batches=max_batches)
        return self.compute_correlation(H, method=method, detrend=detrend)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        corr: Optional[np.ndarray] = None,
        tag: str = "tgcn",
    ) -> dict[str, Path]:
        if corr is None:
            if self._corr_matrix is None:
                raise RuntimeError("Gọi compute_correlation() trước.")
            corr = self._corr_matrix

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        base     = self.results_dir / f"correlation_{tag}_{ts}"
        npz_path = base.with_suffix(".npz")
        csv_path = base.with_suffix(".csv")

        save_dict: dict[str, np.ndarray] = {"correlation_matrix": corr}
        if self._mean_hidden_matrix is not None:
            save_dict["mean_hidden_matrix"] = self._mean_hidden_matrix
        np.savez_compressed(str(npz_path), **save_dict)
        logger.info(f"Saved NPZ → {npz_path}")

        import csv
        N = corr.shape[0]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + [f"node_{i}" for i in range(N)])
            for i in range(N):
                writer.writerow([f"node_{i}"] + [f"{corr[i, j]:.4f}" for j in range(N)])
        logger.info(f"Saved CSV  → {csv_path}")

        return {"npz": npz_path, "csv": csv_path}

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(
        self,
        corr: Optional[np.ndarray] = None,
        tag: str = "tgcn",
        figsize: tuple[int, int] = (12, 10),
        cmap: str = "RdBu_r",
        vmin: float = -1.0,
        vmax: float = 1.0,
        title: Optional[str] = None,
    ) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available — skipping plot.")
            return None

        if corr is None:
            if self._corr_matrix is None:
                raise RuntimeError("Gọi compute_correlation() trước.")
            corr = self._corr_matrix

        N        = corr.shape[0]
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = self.results_dir / f"heatmap_{tag}_{ts}.png"

        fig, ax = plt.subplots(figsize=figsize)
        im  = ax.imshow(corr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r", fontsize=11)

        _title = title or f"T-GCN Node Correlation Matrix (fixed v2) — {N} nodes"
        ax.set_title(_title, fontsize=13, pad=12)
        ax.set_xlabel("Node index", fontsize=11)
        ax.set_ylabel("Node index", fontsize=11)

        if N <= 50:
            ax.set_xticks(range(N))
            ax.set_yticks(range(N))
            ax.set_xticklabels([str(i) for i in range(N)], rotation=90, fontsize=7)
            ax.set_yticklabels([str(i) for i in range(N)], fontsize=7)
        else:
            step   = max(1, N // 20)
            ticks  = list(range(0, N, step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks], rotation=45, fontsize=8)
            ax.set_yticklabels([str(t) for t in ticks], fontsize=8)

        off_diag = corr[~np.eye(N, dtype=bool)]
        stats_txt = (
            f"mean={off_diag.mean():.3f}  "
            f"std={off_diag.std():.3f}  "
            f"max={off_diag.max():.3f}  "
            f"min={off_diag.min():.3f}"
        )
        fig.text(0.5, 0.01, stats_txt, ha="center", fontsize=9, color="gray")

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved PNG  → {png_path}")
        return png_path

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def correlation_matrix(self) -> Optional[np.ndarray]:
        return self._corr_matrix

    @property
    def hidden_matrix(self) -> Optional[np.ndarray]:
        return self._hidden_matrix