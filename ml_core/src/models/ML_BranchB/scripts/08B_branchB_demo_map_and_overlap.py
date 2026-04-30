# ml_core/src/models/ML_BranchB/scripts/08B_branchB_demo_map_and_overlap.py
"""
Branch B — demo map + Top-K overlap for all Rt/Gt prediction methods.

What this script does:
1) Computes Top-K directed-edge overlap between each predicted Rt/Gt method and True-Rt.
2) Supports all Branch-B Rt/Gt methods:
   - true_gt
   - persistence_gt
   - ewma_gt
   - sparse_tvpvar_gt
   - factorized_var_gt
   - factorized_mar_gt
   - factorized_tvpvar_gt
   - dense_tvpvar_gt, only recommended with small --max-nodes
3) Draws a real OSM-based folium map using matched OSM edge metadata:
   outputs/branchA/match_summary/matched_osm_edge_metadata.csv

Important notes:
- no_gt / No-Rt has no relation matrix, so it is skipped in overlap/map.
- Node in Branch A/B = matched OSM directed edge. We draw each relation
  source_node -> target_node by connecting the midpoints of the corresponding
  real OSM directed edges, and also highlighting the source/target road edges.
- For speed, use --max-nodes 512 while testing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Avoid CPU oversubscription when numpy/sklearn operations are used inside
# factorized methods.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import folium
except Exception:
    folium = None


EPS = 1e-8

METHOD_TO_SCRIPT = {
    "true_gt": "06_branchB_run_xt_forecast_true_gt.py",
    "persistence_gt": "06_branchB_run_xt_forecast_persistence_gt.py",
    "ewma_gt": "06_branchB_run_xt_forecast_ewma_gt.py",
    "sparse_tvpvar_gt": "06_branchB_run_xt_forecast_sparse_tvpvar_gt.py",
    "factorized_var_gt": "06_branchB_run_xt_forecast_factorized_var_gt.py",
    "factorized_mar_gt": "06_branchB_run_xt_forecast_factorized_mar_gt.py",
    "factorized_tvpvar_gt": "06_branchB_run_xt_forecast_factorized_tvpvar_gt.py",
    "dense_tvpvar_gt": "06_branchB_run_xt_forecast_dense_tvpvar_gt.py",
}

PRACTICAL_G_METHODS = [
    "true_gt",
    "persistence_gt",
    "ewma_gt",
    "sparse_tvpvar_gt",
    "factorized_var_gt",
    "factorized_mar_gt",
    "factorized_tvpvar_gt",
]

ALL_WITH_DENSE_METHODS = PRACTICAL_G_METHODS + ["dense_tvpvar_gt"]

METHOD_COLORS = {
    "true_gt": "black",
    "persistence_gt": "red",
    "ewma_gt": "blue",
    "sparse_tvpvar_gt": "purple",
    "factorized_var_gt": "orange",
    "factorized_mar_gt": "green",
    "factorized_tvpvar_gt": "darkred",
    "dense_tvpvar_gt": "cadetblue",
}

METHOD_LABELS = {
    "true_gt": "True-Rt",
    "persistence_gt": "Persistence-Rt",
    "ewma_gt": "EWMA-Rt",
    "sparse_tvpvar_gt": "Sparse TVP-VAR-Rt",
    "factorized_var_gt": "Factorized VAR-Rt",
    "factorized_mar_gt": "Factorized MAR-Rt",
    "factorized_tvpvar_gt": "Factorized TVP-VAR-Rt",
    "dense_tvpvar_gt": "Dense TVP-VAR-Rt",
}


# ============================================================
# Utils
# ============================================================
def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def parse_int_list(s: Optional[str]) -> List[int]:
    if s is None:
        return []
    out: List[int] = []
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


def parse_str_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def expand_methods(methods_arg: str, include_dense: bool = False) -> List[str]:
    tokens = parse_str_list(methods_arg) or ["all"]
    out: List[str] = []
    skipped: List[str] = []

    for token in tokens:
        if token == "all":
            out.extend(PRACTICAL_G_METHODS)
        elif token == "all_with_dense":
            out.extend(ALL_WITH_DENSE_METHODS)
        elif token == "baselines":
            out.extend(["true_gt", "persistence_gt", "ewma_gt"])
        elif token == "no_gt":
            skipped.append(token)
        elif token in METHOD_TO_SCRIPT:
            out.append(token)
        else:
            raise ValueError(
                f"Unknown method '{token}'. Valid methods: {sorted(METHOD_TO_SCRIPT)} plus all, all_with_dense, baselines. "
                "no_gt is accepted but skipped because it has no Rt/Gt matrix."
            )

    if include_dense and "dense_tvpvar_gt" not in out:
        out.append("dense_tvpvar_gt")

    if skipped:
        log("[WARN] Skipping no_gt because No-Rt has no relation matrix for overlap/map.")

    seen = set()
    final = []
    for m in out:
        if m not in seen:
            seen.add(m)
            final.append(m)
    return final


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML"), Path("/kaggle/working")]
    for p in candidates:
        if (p / "ml_core").exists() and (p / "dataset").exists():
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


def load_module_definitions_only(path: Path):
    """
    Load definitions from standalone 06 scripts without executing their bottom run block.
    """
    text = path.read_text(encoding="utf-8")
    cut_positions = []
    for marker in [
        "\n# -------------------------\n# Run",
        "\nMETHOD_NAME =",
        "\nPROJECT_ROOT = find_project_root()",
        "\nif __name__ == \"__main__\":",
    ]:
        idx = text.find(marker)
        if idx >= 0:
            cut_positions.append(idx)
    if cut_positions:
        text = text[: min(cut_positions)]

    module = types.ModuleType(path.stem)
    module.__file__ = str(path)
    code = compile(text, str(path), "exec")
    exec(code, module.__dict__)
    return module


def pairs_for_horizon(module, meta: pd.DataFrame, horizon: int) -> List[Tuple[int, int]]:
    if hasattr(module, "iter_eval_pairs"):
        return list(module.iter_eval_pairs(meta, int(horizon)))
    T = len(meta)
    return [(i, i + int(horizon)) for i in range(max(0, T - int(horizon)))]


# ============================================================
# Load Branch-B data and node selection
# ============================================================
def load_split(common_dir: Path, split: str) -> Dict[str, Any]:
    d = common_dir / split
    required = [
        d / "G_weight_series.npy",
        d / "segment_ids.npy",
        d / "timestamps.npy",
        d / "G_series_meta.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Branch-B files:\n" + "\n".join(map(str, missing)))

    G = np.load(d / "G_weight_series.npy", mmap_mode="r")
    segment_ids = np.asarray(np.load(d / "segment_ids.npy"), dtype=np.int64)
    timestamps = pd.to_datetime(np.load(d / "timestamps.npy"))
    meta = pd.read_csv(d / "G_series_meta.csv")
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])

    return {
        "G_weight_series": G,
        "segment_ids": segment_ids,
        "timestamps": timestamps,
        "meta": meta,
    }


def subset_split_data(data: Dict[str, Any], node_idx: Optional[np.ndarray]) -> Dict[str, Any]:
    if node_idx is None:
        return data
    idx = np.asarray(node_idx, dtype=np.int64)
    out = dict(data)
    out["segment_ids"] = np.asarray(data["segment_ids"], dtype=np.int64)[idx]
    # Copy only the selected submatrix into RAM; intended for quick tests such as --max-nodes 512.
    out["G_weight_series"] = np.asarray(data["G_weight_series"][:, idx, :][:, :, idx], dtype=np.float32)
    return out


def resolve_node_indices(
    common_dir: Path,
    max_nodes: int = 0,
    node_indices_arg: Optional[str] = None,
    node_ids_arg: Optional[str] = None,
    node_sample: str = "first",
    seed: int = 42,
) -> Optional[np.ndarray]:
    if not node_indices_arg and not node_ids_arg and int(max_nodes) <= 0:
        return None

    seg_path = common_dir / "train" / "segment_ids.npy"
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing segment_ids for node selection: {seg_path}")
    segment_ids = np.asarray(np.load(seg_path), dtype=np.int64)
    N = int(len(segment_ids))

    selected: Optional[np.ndarray] = None

    if node_indices_arg:
        idx = np.asarray(parse_int_list(node_indices_arg), dtype=np.int64)
        if len(idx) == 0:
            raise ValueError("--node-indices was provided but no valid index was parsed.")
        if idx.min() < 0 or idx.max() >= N:
            raise ValueError(f"node index out of range. N={N}, min={idx.min()}, max={idx.max()}")
        selected = idx

    if node_ids_arg:
        requested_ids = np.asarray(parse_int_list(node_ids_arg), dtype=np.int64)
        id_to_pos = {int(v): i for i, v in enumerate(segment_ids)}
        missing = [int(x) for x in requested_ids if int(x) not in id_to_pos]
        if missing:
            raise ValueError(f"Some --node-ids are not in train/segment_ids.npy: {missing[:20]}")
        idx = np.asarray([id_to_pos[int(x)] for x in requested_ids], dtype=np.int64)
        selected = idx if selected is None else np.intersect1d(selected, idx)

    if selected is None:
        max_nodes = min(int(max_nodes), N)
        if max_nodes <= 0 or max_nodes >= N:
            return None
        if node_sample == "first":
            selected = np.arange(max_nodes, dtype=np.int64)
        elif node_sample == "random":
            rng = np.random.default_rng(int(seed))
            selected = np.sort(rng.choice(N, size=max_nodes, replace=False).astype(np.int64))
        else:
            raise ValueError("--node-sample must be first or random")
    else:
        selected = np.asarray(sorted(set(map(int, selected.tolist()))), dtype=np.int64)
        if int(max_nodes) > 0 and len(selected) > int(max_nodes):
            selected = selected[: int(max_nodes)]

    if len(selected) == 0:
        raise ValueError("Node selection is empty.")
    return selected


# ============================================================
# G model building and prediction
# ============================================================
def load_method_module(scripts_dir: Path, method: str):
    script_path = scripts_dir / METHOD_TO_SCRIPT[method]
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script for method {method}: {script_path}")
    return load_module_definitions_only(script_path)


def build_method_models(
    scripts_dir: Path,
    methods: Sequence[str],
    train: Dict[str, Any],
    val: Dict[str, Any],
    test: Dict[str, Any],
    rank: int,
    stop_on_error: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    modules: Dict[str, Any] = {}
    models: Dict[str, Dict[str, Any]] = {}

    for method in methods:
        log("=" * 88)
        log(f"BUILD G MODEL: {method}")
        module = load_method_module(scripts_dir, method)

        # Keep rank controllable for factorized methods if the base script exposes DEFAULT_RANK.
        if hasattr(module, "DEFAULT_RANK"):
            module.DEFAULT_RANK = int(rank)

        try:
            g_model = module.build_g_model(method, train, val, test)
            modules[method] = module
            models[method] = g_model
            log(f"DONE G MODEL: {method}")
        except Exception as e:
            log(f"[ERROR] Failed to build G model for {method}: {e}")
            traceback.print_exc()
            if stop_on_error:
                raise
            log(f"[WARN] Skip method {method} and continue.")

    return modules, models


def predict_G_for_method(
    method: str,
    modules: Dict[str, Any],
    models: Dict[str, Dict[str, Any]],
    split_name: str,
    split_data: Dict[str, Any],
    origin_idx: int,
    target_idx: int,
    horizon: int,
) -> np.ndarray:
    module = modules[method]
    g_model = models[method]
    G_pred = module.predict_G_method(
        method,
        g_model,
        split_name,
        split_data,
        int(origin_idx),
        int(target_idx),
        int(horizon),
    )
    return np.asarray(G_pred, dtype=np.float32)


# ============================================================
# Top-K and overlap
# ============================================================
def topk_edge_set(G: np.ndarray, k: int = 20) -> set:
    """
    G convention: target x source.
    Set item: (source, target).
    """
    G = np.asarray(G, dtype=np.float32)
    N = int(G.shape[0])
    if N <= 1:
        return set()
    k = min(int(k), N - 1)

    absG = np.abs(G).astype(np.float32, copy=True)
    diag = np.arange(N)
    absG[diag, diag] = -np.inf
    idx = np.argpartition(absG, -k, axis=1)[:, -k:]

    out = set()
    for target in range(N):
        for source in idx[target]:
            w = float(G[target, source])
            if np.isfinite(w) and w != 0.0:
                out.add((int(source), int(target)))
    return out


def topk_edges_sorted(G: np.ndarray, k: int = 20, max_edges: int = 200) -> List[Tuple[int, int, float]]:
    """
    Return strongest relations after per-row Top-K.
    G convention: target x source.
    Edge returned: source -> target.
    """
    G = np.asarray(G, dtype=np.float32)
    N = int(G.shape[0])
    if N <= 1:
        return []
    k = min(int(k), N - 1)

    absG = np.abs(G).astype(np.float32, copy=True)
    diag = np.arange(N)
    absG[diag, diag] = -np.inf
    idx = np.argpartition(absG, -k, axis=1)[:, -k:]

    edges: List[Tuple[int, int, float]] = []
    for target in range(N):
        for source in idx[target]:
            w = float(G[target, source])
            if np.isfinite(w) and w != 0.0:
                edges.append((int(source), int(target), w))

    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    if int(max_edges) > 0:
        edges = edges[: int(max_edges)]
    return edges


def overlap_ratio_from_sets(true_set: set, pred_set: set) -> float:
    if not true_set:
        return 0.0
    return float(len(true_set & pred_set) / (len(true_set) + EPS))


# ============================================================
# Plot overlap
# ============================================================
def plot_overlap(df: pd.DataFrame, out_dir: Path) -> None:
    csv_path = out_dir / "branchB_demo_overlap_metrics.csv"
    df.to_csv(csv_path, index=False)
    log(f"Saved: {csv_path}")

    if df.empty:
        log("[WARN] Empty overlap DataFrame. Skip plots.")
        return

    for split in sorted(df["split"].dropna().unique()):
        sub = df[df["split"] == split].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(12, 6))
        for method, g in sub.groupby("method"):
            g = g.sort_values("lag")
            label = METHOD_LABELS.get(str(method), str(method))
            plt.plot(g["lag"], g["overlap_ratio"], marker="o", label=label)

        plt.title(f"Top-K Directed Edge Overlap vs True-Rt | split={split}")
        plt.xlabel("Forecast horizon / lag")
        plt.ylabel("Overlap ratio")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fig_path = out_dir / f"overlap_ratio_{split}.png"
        plt.savefig(fig_path, dpi=180)
        plt.close()
        log(f"Saved: {fig_path}")


# ============================================================
# Real OSM edge metadata and folium map
# ============================================================
def find_edge_meta_csv(project_root: Path, explicit_path: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit_path:
        p = Path(explicit_path)
        candidates.append(p if p.is_absolute() else project_root / p)

    candidates.extend([
        project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchA" / "match_summary" / "matched_osm_edge_metadata.csv",
        project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchA" / "osm_edge_forecasting_dataset" / "tables" / "node_quality.csv",
    ])

    for p in candidates:
        if p.exists():
            return p
    return None


def load_edge_metadata(project_root: Path, explicit_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    path = find_edge_meta_csv(project_root, explicit_path)
    if path is None:
        log("[WARN] Cannot find OSM edge metadata CSV. Map will fall back to synthetic layout.")
        return None

    df = pd.read_csv(path)
    required = ["model_node_id", "u_lat", "u_lon", "v_lat", "v_lon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log(f"[WARN] Edge metadata found but missing columns {missing}: {path}")
        return None

    for c in ["model_node_id", "u_lat", "u_lon", "v_lat", "v_lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["model_node_id", "u_lat", "u_lon", "v_lat", "v_lon"]).copy()
    df["model_node_id"] = df["model_node_id"].astype(np.int64)

    if "mid_lat" not in df.columns:
        df["mid_lat"] = (df["u_lat"] + df["v_lat"]) / 2.0
    if "mid_lon" not in df.columns:
        df["mid_lon"] = (df["u_lon"] + df["v_lon"]) / 2.0

    log(f"Loaded OSM edge metadata: {path} | rows={len(df):,}")
    return df


def make_edge_lookup(edge_meta: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for row in edge_meta.to_dict("records"):
        lookup[int(row["model_node_id"])] = row
    return lookup


def edge_road_coords(row: Dict[str, Any]) -> List[Tuple[float, float]]:
    return [
        (float(row["u_lat"]), float(row["u_lon"])),
        (float(row["v_lat"]), float(row["v_lon"])),
    ]


def edge_mid(row: Dict[str, Any]) -> Tuple[float, float]:
    return (float(row.get("mid_lat", (row["u_lat"] + row["v_lat"]) / 2.0)),
            float(row.get("mid_lon", (row["u_lon"] + row["v_lon"]) / 2.0)))


def tooltip_for_relation(method: str, source_local: int, target_local: int, source_id: int, target_id: int, w: float, lookup: Dict[int, Dict[str, Any]]) -> str:
    s = lookup.get(int(source_id), {})
    t = lookup.get(int(target_id), {})
    s_edge = str(s.get("osm_edge_id", ""))
    t_edge = str(t.get("osm_edge_id", ""))
    s_street = str(s.get("street_names", ""))
    t_street = str(t.get("street_names", ""))
    return (
        f"{METHOD_LABELS.get(method, method)}<br>"
        f"weight={w:.4f}<br>"
        f"source local={source_local}, model_node_id={source_id}, osm_edge={s_edge}<br>"
        f"source street={s_street}<br>"
        f"target local={target_local}, model_node_id={target_id}, osm_edge={t_edge}<br>"
        f"target street={t_street}"
    )


def plot_synthetic_map(edges_true: List[Tuple[int, int, float]], method_edges: Dict[str, list], out_path: Path) -> None:
    if folium is None:
        log("[WARN] folium not installed. Skip map.")
        return

    m = folium.Map(location=[10.77, 106.66], zoom_start=12, tiles="OpenStreetMap")

    def node_xy(idx: int):
        lat = 10.77 + (idx % 100) * 0.00035
        lon = 106.66 + (idx // 100) * 0.00035
        return lat, lon

    fg_true = folium.FeatureGroup(name="True-Rt synthetic", show=True)
    for source, target, w in edges_true:
        folium.PolyLine(
            locations=[node_xy(source), node_xy(target)],
            color="black",
            weight=1,
            opacity=0.35,
            tooltip=f"TRUE {source}->{target}, w={w:.3f}",
        ).add_to(fg_true)
    fg_true.add_to(m)

    for method, edges in method_edges.items():
        fg = folium.FeatureGroup(name=f"{METHOD_LABELS.get(method, method)} synthetic", show=False)
        color = METHOD_COLORS.get(method, "purple")
        for source, target, w in edges:
            folium.PolyLine(
                locations=[node_xy(source), node_xy(target)],
                color=color,
                weight=2,
                opacity=0.45,
                tooltip=f"{method} {source}->{target}, w={w:.3f}",
            ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_path))
    log(f"Saved synthetic map: {out_path}")


def plot_osm_demo_map(
    edges_true: List[Tuple[int, int, float]],
    method_edges: Dict[str, list],
    segment_ids: np.ndarray,
    edge_meta: Optional[pd.DataFrame],
    out_path: Path,
    max_relations: int = 120,
    draw_base_network: bool = True,
    base_road_limit: int = 6000,
) -> None:
    if folium is None:
        log("[WARN] folium not installed. Skip map.")
        return

    if edge_meta is None or edge_meta.empty:
        log("[WARN] No valid OSM geometry. Using synthetic fallback map.")
        plot_synthetic_map(edges_true[:max_relations], {k: v[:max_relations] for k, v in method_edges.items()}, out_path)
        return

    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    lookup = make_edge_lookup(edge_meta)
    available_ids = set(lookup.keys())

    # Center map on selected segment IDs that have geometry.
    selected_ids = [int(x) for x in segment_ids if int(x) in available_ids]
    center_df = edge_meta[edge_meta["model_node_id"].isin(selected_ids)] if selected_ids else edge_meta
    center_lat = float(center_df["mid_lat"].mean())
    center_lon = float(center_df["mid_lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    if draw_base_network:
        fg_base = folium.FeatureGroup(name="OSM matched road edges", show=True)
        base_df = center_df.copy()
        if int(base_road_limit) > 0 and len(base_df) > int(base_road_limit):
            base_df = base_df.head(int(base_road_limit))
        for row in base_df.to_dict("records"):
            folium.PolyLine(
                locations=edge_road_coords(row),
                color="gray",
                weight=1,
                opacity=0.25,
                tooltip=f"OSM edge {row.get('osm_edge_id', '')} | model_node_id={int(row['model_node_id'])}",
            ).add_to(fg_base)
        fg_base.add_to(m)

    def draw_relation_group(method: str, edges: List[Tuple[int, int, float]], show: bool) -> None:
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, "purple")
        fg = folium.FeatureGroup(name=f"{label} top relations", show=show)

        drawn = 0
        for source_local, target_local, w in edges[: int(max_relations)]:
            if source_local >= len(segment_ids) or target_local >= len(segment_ids):
                continue
            source_id = int(segment_ids[source_local])
            target_id = int(segment_ids[target_local])
            source_row = lookup.get(source_id)
            target_row = lookup.get(target_id)
            if source_row is None or target_row is None:
                continue

            tip = tooltip_for_relation(method, source_local, target_local, source_id, target_id, w, lookup)

            # Highlight source and target road segments.
            folium.PolyLine(
                locations=edge_road_coords(source_row),
                color=color,
                weight=3,
                opacity=0.45,
                tooltip="SOURCE road<br>" + tip,
            ).add_to(fg)
            folium.PolyLine(
                locations=edge_road_coords(target_row),
                color=color,
                weight=5,
                opacity=0.70,
                tooltip="TARGET road<br>" + tip,
            ).add_to(fg)

            # Draw dashed relation line from source-edge midpoint to target-edge midpoint.
            folium.PolyLine(
                locations=[edge_mid(source_row), edge_mid(target_row)],
                color=color,
                weight=2,
                opacity=0.40,
                dash_array="6,8",
                tooltip="RELATION source -> target<br>" + tip,
            ).add_to(fg)
            drawn += 1

        log(f"Map layer {method}: drawn_relations={drawn}/{min(len(edges), int(max_relations))}")
        fg.add_to(m)

    # True-Rt reference is shown by default; predicted methods are toggleable.
    draw_relation_group("true_gt", edges_true, show=True)
    for method, edges in method_edges.items():
        # true_gt may also be requested as a method; do not duplicate the layer.
        if method == "true_gt":
            continue
        draw_relation_group(method, edges, show=False)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_path))
    log(f"Saved OSM map: {out_path}")


# ============================================================
# Main
# ============================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--lags", type=str, default="1-9")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--samples-per-split-lag", type=int, default=10)
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Use all, all_with_dense, baselines, or comma-separated method names. no_gt is skipped.",
    )
    parser.add_argument("--include-dense", action="store_true", help="Add dense_tvpvar_gt. Recommended only with --max-nodes.")
    parser.add_argument("--demo-split", type=str, default="test")
    parser.add_argument("--demo-lag", type=int, default=1)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--rank", type=int, default=12, help="Rank for factorized methods if base script exposes DEFAULT_RANK.")

    parser.add_argument("--data-dir", type=str, default=None, help="Prepared Branch-B data dir.")
    parser.add_argument("--edge-meta-csv", type=str, default=None, help="Optional path to matched_osm_edge_metadata.csv.")

    parser.add_argument("--max-nodes", type=int, default=0, help="Use first/random N nodes for quick test. 0 = full.")
    parser.add_argument("--node-indices", type=str, default=None, help="Explicit node positions, e.g. 0-511 or 0,10,20.")
    parser.add_argument("--node-ids", type=str, default=None, help="Explicit model_node_ids from segment_ids.npy.")
    parser.add_argument("--node-sample", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--map-max-relations", type=int, default=120, help="Maximum strongest relations to draw per method layer.")
    parser.add_argument("--base-road-limit", type=int, default=6000, help="Maximum base OSM matched edges to draw. <=0 means all.")
    parser.add_argument("--no-base-network", action="store_true", help="Do not draw gray OSM base network layer.")
    parser.add_argument("--no-overlap", action="store_true", help="Skip overlap computation and only draw demo map.")
    parser.add_argument("--no-map", action="store_true", help="Skip folium map generation and only compute overlap.")
    parser.add_argument("--stop-on-error", action="store_true")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    project_root = find_project_root()
    scripts_dir = project_root / "ml_core" / "src" / "models" / "ML_BranchB" / "scripts"
    out_dir = ensure_dir(project_root / "ml_core" / "src" / "models" / "ML_BranchB" / "results" / "08B_demo_map_overlap")

    if args.data_dir is None:
        common_dir = project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchB" / "osm_edge_gt_like_branchA"
    else:
        common_dir = Path(args.data_dir)
        if not common_dir.is_absolute():
            common_dir = project_root / common_dir

    methods = expand_methods(args.methods, include_dense=bool(args.include_dense))
    if not methods:
        raise ValueError("No valid Rt/Gt methods to run. no_gt is not supported for overlap/map.")

    splits = parse_str_list(args.splits)
    lags = parse_int_list(args.lags)
    node_idx = resolve_node_indices(
        common_dir=common_dir,
        max_nodes=int(args.max_nodes),
        node_indices_arg=args.node_indices,
        node_ids_arg=args.node_ids,
        node_sample=args.node_sample,
        seed=int(args.seed),
    )

    log("PROJECT_ROOT : " + str(project_root))
    log("SCRIPTS_DIR  : " + str(scripts_dir))
    log("COMMON_DIR   : " + str(common_dir))
    log("OUT_DIR      : " + str(out_dir))
    log("SPLITS       : " + str(splits))
    log("LAGS         : " + str(lags))
    log("METHODS      : " + str(methods))
    log("TOPK         : " + str(args.topk))
    log("NODE MODE    : " + ("full" if node_idx is None else f"subset n={len(node_idx)}"))
    log("MAP MAX REL. : " + str(args.map_max_relations))

    # Load and subset splits.
    loaded = {split: subset_split_data(load_split(common_dir, split), node_idx) for split in sorted(set(splits + [args.demo_split]))}
    train = subset_split_data(load_split(common_dir, "train"), node_idx)
    val = loaded.get("val") or subset_split_data(load_split(common_dir, "val"), node_idx)
    test = loaded.get("test") or subset_split_data(load_split(common_dir, "test"), node_idx)

    if not np.array_equal(train["segment_ids"], val["segment_ids"]):
        raise ValueError("train and val segment_ids differ.")
    if not np.array_equal(train["segment_ids"], test["segment_ids"]):
        raise ValueError("train and test segment_ids differ.")

    # Use true_gt module as helper for iter_eval_pairs.
    helper_module = load_method_module(scripts_dir, "true_gt")

    modules, models = build_method_models(
        scripts_dir=scripts_dir,
        methods=methods,
        train=train,
        val=val,
        test=test,
        rank=int(args.rank),
        stop_on_error=bool(args.stop_on_error),
    )
    methods = [m for m in methods if m in modules and m in models]
    if not methods:
        raise RuntimeError("All requested methods failed to build.")

    # Save run config.
    run_config = vars(args).copy()
    run_config.update({
        "project_root": str(project_root),
        "common_dir": str(common_dir),
        "methods_final": methods,
        "node_mode": "full" if node_idx is None else f"subset n={len(node_idx)}",
    })
    with open(out_dir / "branchB_demo_map_overlap_run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------------
    # Overlap metrics
    # --------------------------------------------------------
    all_rows: List[Dict[str, Any]] = []
    if not args.no_overlap:
        for split in splits:
            data = loaded[split]
            G_series = data["G_weight_series"]
            meta = data["meta"]
            log(f"\n[{split}] T={G_series.shape[0]}, G shape={G_series.shape}")

            for lag in lags:
                pairs = pairs_for_horizon(helper_module, meta, lag)
                if not pairs:
                    log(f"[WARN] split={split}, lag={lag}: no valid origin/target pairs.")
                    continue
                pairs = pairs[: int(args.samples_per_split_lag)]

                # Cache true sets per sampled pair for all methods.
                true_sets: List[set] = []
                for origin_idx, target_idx in pairs:
                    G_true = np.asarray(G_series[target_idx], dtype=np.float32)
                    true_sets.append(topk_edge_set(G_true, k=int(args.topk)))

                for method in methods:
                    vals: List[float] = []
                    for (origin_idx, target_idx), s_true in zip(pairs, true_sets):
                        try:
                            G_pred = predict_G_for_method(method, modules, models, split, data, origin_idx, target_idx, lag)
                            s_pred = topk_edge_set(G_pred, k=int(args.topk))
                            vals.append(overlap_ratio_from_sets(s_true, s_pred))
                        except Exception as e:
                            log(f"[ERROR] overlap failed | split={split} lag={lag} method={method} origin={origin_idx}: {e}")
                            if args.stop_on_error:
                                raise

                    all_rows.append({
                        "split": split,
                        "lag": int(lag),
                        "method": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        "topk": int(args.topk),
                        "n_samples": int(len(vals)),
                        "n_segments": int(len(data["segment_ids"])),
                        "overlap_ratio": float(np.mean(vals)) if vals else np.nan,
                        "overlap_std": float(np.std(vals)) if vals else np.nan,
                    })
                    log(f"overlap | split={split} lag={lag} method={method}: {all_rows[-1]['overlap_ratio']:.4f}")

        df = pd.DataFrame(all_rows)
        plot_overlap(df, out_dir)

    # --------------------------------------------------------
    # Demo OSM map
    # --------------------------------------------------------
    if not args.no_map:
        if args.demo_split in loaded:
            data = loaded[args.demo_split]
        else:
            data = subset_split_data(load_split(common_dir, args.demo_split), node_idx)

        G_series = data["G_weight_series"]
        meta = data["meta"]
        demo_lag = int(args.demo_lag)
        pairs = pairs_for_horizon(helper_module, meta, demo_lag)
        if not pairs:
            log(f"[WARN] Cannot create demo map: no pairs for demo_lag={demo_lag}")
            return

        demo_pos = max(0, min(int(args.demo_index), len(pairs) - 1))
        origin_idx, target_idx = pairs[demo_pos]

        log("\nDEMO MAP")
        log(f"split      : {args.demo_split}")
        log(f"origin_idx : {origin_idx}")
        log(f"target_idx : {target_idx}")
        log(f"demo_lag   : {demo_lag}")
        log(f"n_segments : {len(data['segment_ids'])}")

        G_true = np.asarray(G_series[target_idx], dtype=np.float32)
        edges_true = topk_edges_sorted(G_true, k=int(args.topk), max_edges=int(args.map_max_relations))

        method_edges: Dict[str, list] = {}
        for method in methods:
            try:
                G_pred = predict_G_for_method(method, modules, models, args.demo_split, data, origin_idx, target_idx, demo_lag)
                method_edges[method] = topk_edges_sorted(G_pred, k=int(args.topk), max_edges=int(args.map_max_relations))
            except Exception as e:
                log(f"[ERROR] map prediction failed for {method}: {e}")
                if args.stop_on_error:
                    raise

        edge_meta = load_edge_metadata(project_root, args.edge_meta_csv)
        node_tag = "full" if node_idx is None else f"nodes{len(node_idx)}"
        map_path = out_dir / f"demo_osm_map_{args.demo_split}_lag{demo_lag}_idx{origin_idx}_topk{args.topk}_{node_tag}.html"
        plot_osm_demo_map(
            edges_true=edges_true,
            method_edges=method_edges,
            segment_ids=np.asarray(data["segment_ids"], dtype=np.int64),
            edge_meta=edge_meta,
            out_path=map_path,
            max_relations=int(args.map_max_relations),
            draw_base_network=not bool(args.no_base_network),
            base_road_limit=int(args.base_road_limit),
        )

    log("DONE.")


if __name__ == "__main__":
    main()
