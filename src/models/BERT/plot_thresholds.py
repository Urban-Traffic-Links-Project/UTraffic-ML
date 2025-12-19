# scripts/plot_thresholds.py
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Distance
# ============================================================

EARTH_RADIUS_M = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(EARTH_RADIUS_M * c)


# ============================================================
# Paper-style Wpos / tau_pos
# ============================================================

def xcorr_wpos_tau_paper(x: np.ndarray, y: np.ndarray, tau_max: int, eps: float = 1e-6) -> Tuple[float, int]:
    """
    Paper-style:
      - Compute X_{i,j}(tau) Pearson correlation for tau in [-tau_max, +tau_max]
      - tau_pos = argmax X(tau)
      - Wpos = (max(X) - mean(X)) / std(X)
      - clip Wpos at >=0
    """
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    T = x.shape[0]
    if T < 4:
        return 0.0, 0

    taus = np.arange(-tau_max, tau_max + 1, dtype=np.int32)
    Xtaus = np.zeros((taus.size,), dtype=np.float32)

    for k, tau in enumerate(taus):
        if abs(tau) >= T:
            Xtaus[k] = 0.0
            continue

        if tau < 0:
            xa = x[-tau:]
            ya = y[: T + tau]
        elif tau > 0:
            xa = x[: T - tau]
            ya = y[tau:]
        else:
            xa = x
            ya = y

        if xa.size < 4:
            Xtaus[k] = 0.0
            continue

        xa0 = xa - xa.mean()
        ya0 = ya - ya.mean()
        denom = (np.sqrt((xa0 * xa0).sum()) * np.sqrt((ya0 * ya0).sum())) + eps
        Xtaus[k] = float((xa0 * ya0).sum() / denom)

    idx = int(np.argmax(Xtaus))
    maxX = float(Xtaus[idx])
    meanX = float(Xtaus.mean())
    stdX = float(Xtaus.std()) + eps

    Wpos = (maxX - meanX) / stdX
    if Wpos < 0:
        Wpos = 0.0
    tau_pos = int(taus[idx])
    return float(Wpos), tau_pos


# ============================================================
# Hourly shuffle (paper-style null)
# ============================================================

def hourly_shuffle_by_hourbin(x: np.ndarray, tod_minutes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Shuffle values within each hour-of-day bin (0..23) independently.
    Keeps diurnal distribution but destroys temporal order (lead-lag).
    """
    x = x.copy()
    hours = (tod_minutes // 60).astype(np.int32)
    for h in range(24):
        idx = np.where(hours == h)[0]
        if idx.size <= 1:
            continue
        perm = rng.permutation(idx.size)
        x[idx] = x[idx][perm]
    return x


# ============================================================
# Centers (midpoint) from segments.csv + nodes.csv
# ============================================================

def build_segment_centers_latlon(segments_csv: str, nodes_csv: str, segment_index_csv: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      segment_ids_sorted: (N,)
      centers_latlon: (N,2) aligned with segment_ids_sorted
    Assumes segments.csv has columns: segment_id, node_u, node_v
            nodes.csv has: node_id, lat, lon
    If segment_index_csv exists, we use it to define the final ordering of N.
    """
    seg = pd.read_csv(segments_csv)
    nodes = pd.read_csv(nodes_csv)

    nodes = nodes.set_index("node_id")[["lat", "lon"]]

    # midpoint (u,v)
    lat_u = nodes.loc[seg["node_u"].values, "lat"].values
    lon_u = nodes.loc[seg["node_u"].values, "lon"].values
    lat_v = nodes.loc[seg["node_v"].values, "lat"].values
    lon_v = nodes.loc[seg["node_v"].values, "lon"].values

    center_lat = (lat_u + lat_v) / 2.0
    center_lon = (lon_u + lon_v) / 2.0

    seg_centers = pd.DataFrame({
        "segment_id": seg["segment_id"].values,
        "lat": center_lat,
        "lon": center_lon
    })

    if segment_index_csv is not None:
        # enforce ordering idx -> segment_id
        idxmap = pd.read_csv(segment_index_csv)
        # expected columns: idx, segment_id
        idxmap = idxmap.sort_values("idx")
        seg_centers = idxmap.merge(seg_centers, on="segment_id", how="left")
        segment_ids_sorted = seg_centers["segment_id"].values.astype(np.int64)
        centers_latlon = seg_centers[["lat", "lon"]].values.astype(np.float32)
        return segment_ids_sorted, centers_latlon

    # fallback: sort by segment_id
    seg_centers = seg_centers.sort_values("segment_id")
    segment_ids_sorted = seg_centers["segment_id"].values.astype(np.int64)
    centers_latlon = seg_centers[["lat", "lon"]].values.astype(np.float32)
    return segment_ids_sorted, centers_latlon


# ============================================================
# Main plotting
# ============================================================

def sample_window(values: np.ndarray, time_of_day: np.ndarray, dates: Optional[np.ndarray],
                  window_len: int, rng: np.random.Generator,
                  fixed_end_t: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Pick one random window of length window_len: [t0..t] and return X_win (L,N), tod_minutes_win (L,), t_end.
    If fixed_end_t is provided, use it.
    """
    T = values.shape[0]
    if fixed_end_t is None:
        t_end = int(rng.integers(window_len - 1, T))
    else:
        t_end = int(fixed_end_t)

    t0 = t_end - window_len + 1
    X = values[t0:t_end+1, :]
    tod_minutes = (time_of_day[t0:t_end+1] * 1440.0).astype(np.int32)
    return X, tod_minutes, t_end


def plot_wpos_vs_distance(
    values: np.ndarray,               # (T,N)
    time_of_day: np.ndarray,           # (T,) in [0,1]
    centers_latlon: np.ndarray,        # (N,2)
    tau_max: int,
    window_len: int,
    num_pairs: int,
    max_x_km: float,
    seed: int = 13,
    Wmin_line: Optional[float] = None,
    Dmin_line_km: Optional[float] = None,
    Dmax_line_km: Optional[float] = None,
    title: str = "",
    fixed_end_t: Optional[int] = None,
    out_png: str = None,
) -> Dict[str, Any]:
    """
    Make a Figure-3-like scatter:
      y-axis: Wpos
      x-axis: D_ij (km)
      blue: original
      green: hourly-shuffled
    """
    rng = np.random.default_rng(seed)
    Xwin, tod_minutes_win, t_end = sample_window(values, time_of_day, None, window_len, rng, fixed_end_t=fixed_end_t)

    L, N = Xwin.shape

    # precompute shuffled window for each node independently
    Xshuf = np.empty_like(Xwin, dtype=np.float32)
    for n in range(N):
        Xshuf[:, n] = hourly_shuffle_by_hourbin(Xwin[:, n], tod_minutes_win, rng)

    # sample random pairs (i<j), filter by distance <= max_x_km
    ii = rng.integers(0, N, size=num_pairs, dtype=np.int64)
    jj = rng.integers(0, N, size=num_pairs, dtype=np.int64)
    keep = (ii != jj)
    ii = ii[keep]
    jj = jj[keep]

    D_km_list: List[float] = []
    W_orig_list: List[float] = []
    W_shuf_list: List[float] = []

    for i, j in zip(ii.tolist(), jj.tolist()):
        lat1, lon1 = centers_latlon[i]
        lat2, lon2 = centers_latlon[j]
        d_km = haversine_m(float(lat1), float(lon1), float(lat2), float(lon2)) / 1000.0
        if d_km > max_x_km:
            continue

        w1, _ = xcorr_wpos_tau_paper(Xwin[:, i], Xwin[:, j], tau_max=tau_max)
        w2, _ = xcorr_wpos_tau_paper(Xshuf[:, i], Xshuf[:, j], tau_max=tau_max)

        D_km_list.append(d_km)
        W_orig_list.append(w1)
        W_shuf_list.append(w2)

    D_km = np.array(D_km_list, dtype=np.float32)
    W_orig = np.array(W_orig_list, dtype=np.float32)
    W_shuf = np.array(W_shuf_list, dtype=np.float32)

    plt.figure(figsize=(7.8, 5.2))
    plt.scatter(D_km, W_orig, s=3, alpha=0.7, label="Original")
    plt.scatter(D_km, W_shuf, s=3, alpha=0.7, label="Hourly-shuffled")
    plt.xlim(0, max_x_km)
    plt.xlabel(r"$D_{ij}$ (km)")
    plt.ylabel(r"$W^{pos}_{ij}$")
    if title:
        plt.title(title)
    else:
        plt.title(f"Wpos vs Distance (window_end={t_end}, L={window_len}, tau_max={tau_max})")

    # Optional threshold lines
    if Wmin_line is not None:
        plt.axhline(float(Wmin_line), linestyle="--", linewidth=1, label=f"Wmin={Wmin_line:.3g}")
    if Dmin_line_km is not None:
        plt.axvline(float(Dmin_line_km), linestyle="--", linewidth=1, label=f"Dmin={Dmin_line_km:.3g} km")
    if Dmax_line_km is not None:
        plt.axvline(float(Dmax_line_km), linestyle="--", linewidth=1, label=f"Dmax={Dmax_line_km:.3g} km")

    plt.legend()
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=200)
    else:
        plt.show()

    return {
        "t_end": t_end,
        "D_km": D_km,
        "W_orig": W_orig,
        "W_shuf": W_shuf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traffic_npz", type=str, required=True)
    ap.add_argument("--segments_csv", type=str, required=True)
    ap.add_argument("--nodes_csv", type=str, required=True)
    ap.add_argument("--segment_index_csv", type=str, default=None)
    ap.add_argument("--tau_max", type=int, default=3, help="lag in steps (15min/step -> tau_max=3 means ±45min)")
    ap.add_argument("--window_len", type=int, default=48, help="history length used to compute Wpos in this plot")
    ap.add_argument("--num_pairs", type=int, default=60000)
    ap.add_argument("--max_x_km", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--Wmin_line", type=float, default=None)
    ap.add_argument("--Dmin_line_km", type=float, default=0)
    ap.add_argument("--Dmax_line_km", type=float, default=5)
    ap.add_argument("--fixed_end_t", type=int, default=None, help="optional fixed window end index t")
    ap.add_argument("--out_png", type=str, default=None)

    args = ap.parse_args()

    z = np.load(args.traffic_npz, allow_pickle=True)
    values = z["values"].astype(np.float32)           # (T,N)
    time_of_day = z["time_of_day"].astype(np.float32) # (T,) in [0,1]

    # Build centers aligned to segment_index order (recommended)
    _, centers_latlon = build_segment_centers_latlon(
        segments_csv=args.segments_csv,
        nodes_csv=args.nodes_csv,
        segment_index_csv=args.segment_index_csv
    )

    plot_wpos_vs_distance(
        values=values,
        time_of_day=time_of_day,
        centers_latlon=centers_latlon,
        tau_max=args.tau_max,
        window_len=args.window_len,
        num_pairs=args.num_pairs,
        max_x_km=args.max_x_km,
        seed=args.seed,
        Wmin_line=args.Wmin_line,
        Dmin_line_km=args.Dmin_line_km,
        Dmax_line_km=args.Dmax_line_km,
        title="Figure-3 style: Wpos vs Dij (Original vs Hourly-shuffled)",
        fixed_end_t=args.fixed_end_t,
        out_png=args.out_png,
    )


if __name__ == "__main__":
    main()
