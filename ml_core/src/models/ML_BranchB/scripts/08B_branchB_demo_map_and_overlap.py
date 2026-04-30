
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import folium
except Exception:
    folium = None


EPS = 1e-8


# ============================================================
# Utils
# ============================================================
def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_int_list(s: str) -> List[int]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML"), Path("/kaggle/working")]
    for p in candidates:
        if (p / "ml_core").exists():
            return p
        if p.name == "UTraffic-ML":
            return p
        if (p / "UTraffic-ML").exists():
            pp = p / "UTraffic-ML"
            if (pp / "ml_core").exists():
                return pp
    return cwd


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================
# Load data
# ============================================================
def load_split(common_dir: Path, split: str) -> Dict[str, object]:
    d = common_dir / split
    G = np.load(d / "G_weight_series.npy", mmap_mode="r")
    seg = np.load(d / "segment_ids.npy")
    timestamps = pd.to_datetime(np.load(d / "timestamps.npy"))
    meta = pd.read_csv(d / "G_series_meta.csv")
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])
    return {
        "G": G,
        "segment_ids": seg.astype(np.int64),
        "timestamps": timestamps,
        "meta": meta,
    }


def valid_origin_indices(T: int, lag: int) -> List[int]:
    return list(range(0, max(0, T - lag)))


# ============================================================
# G prediction methods for demo
# ============================================================
def predict_G(method: str, G_series, origin_idx: int, target_idx: int, ewma_alpha: float = 0.3) -> np.ndarray:
    if method == "true_gt":
        return np.asarray(G_series[target_idx], dtype=np.float32)

    if method == "persistence_gt":
        return np.asarray(G_series[origin_idx], dtype=np.float32)

    if method == "ewma_gt":
        # EWMA only uses history up to origin_idx, no future leakage
        G_ewma = np.asarray(G_series[0], dtype=np.float32)
        for t in range(1, origin_idx + 1):
            G_ewma = ewma_alpha * np.asarray(G_series[t], dtype=np.float32) + (1.0 - ewma_alpha) * G_ewma
        return G_ewma.astype(np.float32)

    raise ValueError(f"Unsupported method: {method}")


# ============================================================
# Top-K and overlap
# ============================================================
def get_topk_edges(G: np.ndarray, k: int = 20) -> List[Tuple[int, int, float]]:
    """
    G convention: target x source.
    Edge returned: source -> target.
    """
    G = np.asarray(G, dtype=np.float32)
    N = G.shape[0]
    k = min(k, max(1, N - 1))

    edges = []

    absG = np.abs(G).copy()
    diag = np.arange(N)
    absG[diag, diag] = -np.inf

    idx = np.argpartition(absG, -k, axis=1)[:, -k:]

    for target in range(N):
        for source in idx[target]:
            w = float(G[target, source])
            if np.isfinite(w) and w != 0.0:
                edges.append((int(source), int(target), w))

    return edges


def overlap_ratio(edges_true, edges_pred) -> float:
    s_true = {(i, j) for i, j, _ in edges_true}
    s_pred = {(i, j) for i, j, _ in edges_pred}
    if not s_true:
        return 0.0
    return len(s_true & s_pred) / (len(s_true) + EPS)


# ============================================================
# Plot overlap
# ============================================================
def plot_overlap(df: pd.DataFrame, out_dir: Path) -> None:
    csv_path = out_dir / "branchB_demo_overlap_metrics.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    for split in sorted(df["split"].unique()):
        sub = df[df["split"] == split].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(10, 5))
        for method, g in sub.groupby("method"):
            g = g.sort_values("lag")
            plt.plot(g["lag"], g["overlap_ratio"], marker="o", label=method)

        plt.title(f"Top-K Directed Edge Overlap vs True-G | split={split}")
        plt.xlabel("Forecast horizon / lag")
        plt.ylabel("Overlap ratio")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fig_path = out_dir / f"overlap_ratio_{split}.png"
        plt.savefig(fig_path, dpi=180)
        plt.close()
        print("Saved:", fig_path)


# ============================================================
# Demo map
# ============================================================
def plot_demo_map(edges_true, method_edges: Dict[str, list], out_path: Path) -> None:
    """
    Fallback demo map: if no real geometry file is available, draw synthetic points.
    This still shows source-target relation structure, but not true OSM geometry.
    """
    if folium is None:
        print("[WARN] folium not installed. Skip map.")
        return

    m = folium.Map(location=[10.77, 106.66], zoom_start=12)

    # synthetic layout only
    def node_xy(idx: int):
        lat = 10.77 + (idx % 100) * 0.00035
        lon = 106.66 + (idx // 100) * 0.00035
        return lat, lon

    # True edges: black
    for source, target, w in edges_true[:200]:
        lat1, lon1 = node_xy(source)
        lat2, lon2 = node_xy(target)
        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color="black",
            weight=1,
            opacity=0.35,
            tooltip=f"TRUE {source}->{target}, w={w:.3f}",
        ).add_to(m)

    colors = {
        "persistence_gt": "red",
        "ewma_gt": "blue",
        "true_gt": "green",
    }

    for method, edges in method_edges.items():
        color = colors.get(method, "purple")
        for source, target, w in edges[:200]:
            lat1, lon1 = node_xy(source)
            lat2, lon2 = node_xy(target)
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color=color,
                weight=2,
                opacity=0.45,
                tooltip=f"{method} {source}->{target}, w={w:.3f}",
            ).add_to(m)

    m.save(str(out_path))
    print("Saved:", out_path)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--lags", type=str, default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--samples-per-split-lag", type=int, default=20)
    parser.add_argument("--methods", type=str, default="persistence_gt,ewma_gt")
    parser.add_argument("--ewma-alpha", type=float, default=0.3)
    parser.add_argument("--demo-split", type=str, default="val")
    parser.add_argument("--demo-lag", type=int, default=1)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--rank", type=int, default=12)  # kept for CLI compatibility, not used here

    args = parser.parse_args()

    project_root = find_project_root()
    common_dir = project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchB" / "osm_edge_gt_like_branchA"
    out_dir = ensure_dir(project_root / "ml_core" / "src" / "models" / "ML_BranchB" / "results" / "08B_demo_map_overlap")

    splits = parse_str_list(args.splits)
    lags = parse_int_list(args.lags)
    methods = parse_str_list(args.methods)

    print("PROJECT_ROOT:", project_root)
    print("COMMON_DIR  :", common_dir)
    print("OUT_DIR     :", out_dir)
    print("SPLITS      :", splits)
    print("LAGS        :", lags)
    print("METHODS     :", methods)
    print("TOPK        :", args.topk)
    print("SAMPLES     :", args.samples_per_split_lag)

    all_rows = []

    loaded = {split: load_split(common_dir, split) for split in splits}

    for split in splits:
        data = loaded[split]
        G_series = data["G"]
        T = G_series.shape[0]

        print(f"\n[{split}] T={T}, G shape={G_series.shape}")

        for lag in lags:
            origins = valid_origin_indices(T, lag)
            if not origins:
                print(f"[WARN] split={split}, lag={lag}: no valid origin.")
                continue

            origins = origins[: args.samples_per_split_lag]

            for method in methods:
                vals = []
                for origin_idx in origins:
                    target_idx = origin_idx + lag

                    G_true = np.asarray(G_series[target_idx], dtype=np.float32)
                    G_pred = predict_G(
                        method=method,
                        G_series=G_series,
                        origin_idx=origin_idx,
                        target_idx=target_idx,
                        ewma_alpha=args.ewma_alpha,
                    )

                    edges_true = get_topk_edges(G_true, k=args.topk)
                    edges_pred = get_topk_edges(G_pred, k=args.topk)

                    vals.append(overlap_ratio(edges_true, edges_pred))

                all_rows.append({
                    "split": split,
                    "lag": lag,
                    "method": method,
                    "topk": args.topk,
                    "n_samples": len(vals),
                    "overlap_ratio": float(np.mean(vals)) if vals else np.nan,
                    "overlap_std": float(np.std(vals)) if vals else np.nan,
                })

    df = pd.DataFrame(all_rows)
    plot_overlap(df, out_dir)

    # demo map
    if args.demo_split in loaded:
        data = loaded[args.demo_split]
    else:
        data = load_split(common_dir, args.demo_split)

    G_series = data["G"]
    T = G_series.shape[0]

    demo_lag = int(args.demo_lag)
    valid = valid_origin_indices(T, demo_lag)

    if not valid:
        print(f"[WARN] Cannot create demo map: T={T}, demo_lag={demo_lag}")
        return

    demo_pos = max(0, min(int(args.demo_index), len(valid) - 1))
    origin_idx = valid[demo_pos]
    target_idx = origin_idx + demo_lag

    print("\nDEMO MAP")
    print("split     :", args.demo_split)
    print("origin_idx:", origin_idx)
    print("target_idx:", target_idx)
    print("T         :", T)

    G_true = np.asarray(G_series[target_idx], dtype=np.float32)
    edges_true = get_topk_edges(G_true, k=args.topk)

    method_edges = {}
    for method in methods:
        G_pred = predict_G(method, G_series, origin_idx, target_idx, ewma_alpha=args.ewma_alpha)
        method_edges[method] = get_topk_edges(G_pred, k=args.topk)

    map_path = out_dir / f"demo_map_{args.demo_split}_lag{demo_lag}_idx{origin_idx}_topk{args.topk}.html"
    plot_demo_map(edges_true, method_edges, map_path)


if __name__ == "__main__":
    main()