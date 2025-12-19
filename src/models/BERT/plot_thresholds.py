# scripts/plot_thresholds.py
from __future__ import annotations
import argparse
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
    T = int(x.shape[0])
    if T < 4:
        return 0.0, 0

    taus = np.arange(-tau_max, tau_max + 1, dtype=np.int32)
    Xtaus = np.zeros((taus.size,), dtype=np.float32)

    for k, tau in enumerate(taus):
        if abs(int(tau)) >= T:
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
# Shuffle within time-of-day bins (paper-style null)
# ============================================================

def shuffle_by_tod_bins(
    x: np.ndarray,                # (L,)
    tod_minutes: np.ndarray,      # (L,) minute-of-day [0..1439]
    rng: np.random.Generator,
    bin_minutes: int = 60,
) -> np.ndarray:
    """
    Shuffle values within each time-of-day bin (bin_minutes).
    - bin_minutes=60  -> "hourly shuffling" (paper wording)
    - bin_minutes=120 -> 2h bins (better for 15min/step)
    - bin_minutes=180 -> 3h bins, etc.
    """
    assert bin_minutes > 0 and (1440 % bin_minutes == 0), "bin_minutes must divide 1440"
    x = x.copy()
    bin_id = (tod_minutes // bin_minutes).astype(np.int32)
    n_bins = 1440 // bin_minutes

    for b in range(n_bins):
        idx = np.where(bin_id == b)[0]
        if idx.size <= 1:
            continue
        perm = rng.permutation(idx.size)
        x[idx] = x[idx][perm]
    return x


# ============================================================
# Segment centers (midpoint)
# ============================================================

def build_segment_centers_latlon(
    segments_csv: str,
    nodes_csv: str,
    segment_index_csv: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      segment_ids_sorted: (N,)
      centers_latlon: (N,2) aligned with segment_ids_sorted
    """
    seg = pd.read_csv(segments_csv)
    nodes = pd.read_csv(nodes_csv).set_index("node_id")[["lat", "lon"]]

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
        idxmap = pd.read_csv(segment_index_csv).sort_values("idx")
        seg_centers = idxmap.merge(seg_centers, on="segment_id", how="left")
        segment_ids_sorted = seg_centers["segment_id"].values.astype(np.int64)
        centers_latlon = seg_centers[["lat", "lon"]].values.astype(np.float32)
        return segment_ids_sorted, centers_latlon

    seg_centers = seg_centers.sort_values("segment_id")
    segment_ids_sorted = seg_centers["segment_id"].values.astype(np.int64)
    centers_latlon = seg_centers[["lat", "lon"]].values.astype(np.float32)
    return segment_ids_sorted, centers_latlon


# ============================================================
# Build concatenated period across multiple days
# ============================================================

def parse_hhmm(s: str) -> int:
    """'07:00' -> 420 (minutes)"""
    hh, mm = s.split(":")
    return int(hh) * 60 + int(mm)

def build_day_to_indices_for_period(
    dates: np.ndarray,        # (T,) 'YYYY-MM-DD'
    tod_minutes: np.ndarray,  # (T,) minute-of-day 0..1439
    start_min: int,
    end_min: int,
) -> Dict[str, np.ndarray]:
    """
    Map date -> sorted indices t within [start_min, end_min).
    """
    out: Dict[str, List[int]] = {}
    for t in range(len(dates)):
        m = int(tod_minutes[t])
        if start_min <= m < end_min:
            d = str(dates[t])
            out.setdefault(d, []).append(int(t))
    return {d: np.array(sorted(idx), dtype=np.int64) for d, idx in out.items()}

def estimate_step_minutes(tod_minutes: np.ndarray, idx: np.ndarray) -> Optional[int]:
    """
    Try to estimate step size (minutes) from one day's indices.
    Return None if cannot.
    """
    if idx.size < 3:
        return None
    m = np.sort(tod_minutes[idx])
    diffs = np.diff(m)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    # most common diff
    vals, cnt = np.unique(diffs, return_counts=True)
    return int(vals[np.argmax(cnt)])

def sample_period_multi_days(
    values: np.ndarray,           # (T,N)
    dates: np.ndarray,            # (T,)
    tod_minutes: np.ndarray,      # (T,)
    rng: np.random.Generator,
    days: int,
    start_min: int,
    duration_min: int,
    fixed_end_date: Optional[str] = None,
    require_full_coverage: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Concatenate same time-of-day period across `days` consecutive dates (in available-date order).
    Returns:
      Xcat: (Lcat,N)
      tod_cat: (Lcat,) minute-of-day
      chosen_days: list[str]
    """
    end_min = start_min + duration_min
    if end_min > 1440:
        raise ValueError("start + duration exceeds 24h; keep within the same day.")

    day2idx = build_day_to_indices_for_period(dates, tod_minutes, start_min, end_min)
    all_days = sorted(day2idx.keys())
    if len(all_days) < days:
        raise ValueError(f"Not enough days for requested period. Have {len(all_days)}, need {days}.")

    # Determine expected steps per day (optional strict check)
    expected_steps = None
    if require_full_coverage:
        # estimate from the first day that has enough points
        for d in all_days:
            step = estimate_step_minutes(tod_minutes, day2idx[d])
            if step is not None and step > 0:
                expected_steps = duration_min // step
                break

    # Filter days by full coverage if required
    if require_full_coverage and expected_steps is not None:
        good_days = [d for d in all_days if day2idx[d].size >= expected_steps]
        all_days = good_days
        if len(all_days) < days:
            raise ValueError(
                f"Not enough fully-covered days. Need {days}, have {len(all_days)}. "
                f"(expected_steps/day ~ {expected_steps})"
            )

    if fixed_end_date is None:
        end_pos = int(rng.integers(days - 1, len(all_days)))
    else:
        if fixed_end_date not in all_days:
            raise ValueError(f"fixed_end_date={fixed_end_date} not present/usable for this period.")
        end_pos = all_days.index(fixed_end_date)

    start_pos = end_pos - days + 1
    chosen_days = all_days[start_pos:end_pos + 1]

    idx_cat = np.concatenate([day2idx[d] for d in chosen_days], axis=0)
    Xcat = values[idx_cat, :].astype(np.float32)
    tod_cat = tod_minutes[idx_cat].astype(np.int32)

    return Xcat, tod_cat, chosen_days


# ============================================================
# Plot
# ============================================================

def plot_wpos_vs_distance(
    values: np.ndarray,               # (T,N)
    dates: np.ndarray,                # (T,)
    time_of_day: np.ndarray,          # (T,) in [0,1]
    centers_latlon: np.ndarray,       # (N,2)
    tau_max: int,
    days: int,
    start_hhmm: str,
    duration_hours: float,
    num_pairs: int,
    max_x_km: float,
    seed: int = 13,
    shuffle_bin_minutes: int = 60,
    require_full_coverage: bool = True,
    fixed_end_date: Optional[str] = None,
    Wmin_line: Optional[float] = None,
    Dmin_line_km: Optional[float] = None,
    Dmax_line_km: Optional[float] = None,
    out_png: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Figure-3-like scatter using multi-day same-period concatenation.
    """
    rng = np.random.default_rng(seed)
    tod_minutes_all = (time_of_day * 1440.0).astype(np.int32)

    start_min = parse_hhmm(start_hhmm)
    duration_min = int(round(duration_hours * 60))

    Xwin, tod_win, chosen_days = sample_period_multi_days(
        values=values,
        dates=dates,
        tod_minutes=tod_minutes_all,
        rng=rng,
        days=days,
        start_min=start_min,
        duration_min=duration_min,
        fixed_end_date=fixed_end_date,
        require_full_coverage=require_full_coverage,
    )
    L, N = Xwin.shape
    print(f"[INFO] Using period start={start_hhmm}, duration={duration_hours}h, days={days}, Lcat={L}")
    print(f"[INFO] chosen_days={chosen_days}")
    print(f"[INFO] shuffle_bin_minutes={shuffle_bin_minutes}")

    # shuffled per node independently
    Xshuf = np.empty_like(Xwin, dtype=np.float32)
    for n in range(N):
        Xshuf[:, n] = shuffle_by_tod_bins(Xwin[:, n], tod_win, rng, bin_minutes=shuffle_bin_minutes)

    # sample pairs
    ii = rng.integers(0, N, size=num_pairs, dtype=np.int64)
    jj = rng.integers(0, N, size=num_pairs, dtype=np.int64)
    keep = (ii != jj)
    ii, jj = ii[keep], jj[keep]

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

    # quick sanity: shuffle changed data?
    if L > 0:
        mad0 = float(np.mean(np.abs(Xshuf[:, 0] - Xwin[:, 0])))
        print(f"[DEBUG] mean abs diff (node0) = {mad0:.6f}")

    plt.figure(figsize=(8.2, 5.4))
    plt.scatter(D_km, W_orig, s=3, alpha=0.7, label="Original")
    plt.scatter(D_km, W_shuf, s=3, alpha=0.7, label=f"Shuffled ({shuffle_bin_minutes}min bins)")
    plt.xlim(0, max_x_km)
    plt.xlabel(r"$D_{ij}$ (km)")
    plt.ylabel(r"$W^{pos}_{ij}$")
    plt.title(f"Wpos vs Dij | {start_hhmm}+{duration_hours}h x {days} days | tau_max={tau_max}")

    if Wmin_line is not None:
        plt.axhline(float(Wmin_line), linestyle="--", linewidth=1, label=f"Wmin={Wmin_line:.3g}")
    if Dmin_line_km is not None:
        plt.axvline(float(Dmin_line_km), linestyle="--", linewidth=1, label=f"Dmin={Dmin_line_km:.3g} km")
    if Dmax_line_km is not None:
        plt.axvline(float(Dmax_line_km), linestyle="--", linewidth=1, label=f"Dmax={Dmax_line_km:.3g} km")

    plt.legend()
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"[INFO] saved: {out_png}")
    else:
        plt.show()

    return {
        "chosen_days": chosen_days,
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

    ap.add_argument("--tau_max", type=int, default=6, help="lag in steps (15min/step: tau_max=6 => ±90min)")
    ap.add_argument("--days", type=int, default=5, help="number of days to concatenate")
    ap.add_argument("--start_hhmm", type=str, default="07:00", help="period start time HH:MM")
    ap.add_argument("--duration_hours", type=float, default=6.0, help="period length in hours (e.g., 6.0)")
    ap.add_argument("--fixed_end_date", type=str, default=None, help="optional YYYY-MM-DD to end at that date")

    ap.add_argument("--shuffle_bin_minutes", type=int, default=60,
                    help="shuffle bin in minutes (paper=60; for 15min step try 120/180)")
    ap.add_argument("--no_full_coverage", action="store_true",
                    help="disable requiring full coverage per day in selected period")

    ap.add_argument("--num_pairs", type=int, default=20000)
    ap.add_argument("--max_x_km", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--Wmin_line", type=float, default=None)
    ap.add_argument("--Dmin_line_km", type=float, default=None)
    ap.add_argument("--Dmax_line_km", type=float, default=None)

    ap.add_argument("--out_png", type=str, default=None)

    args = ap.parse_args()

    z = np.load(args.traffic_npz, allow_pickle=True)
    values = z["values"].astype(np.float32)               # (T,N)
    time_of_day = z["time_of_day"].astype(np.float32)     # (T,)
    if "dates" not in z:
        raise ValueError("traffic_tensor.npz must contain 'dates' for multi-day period concatenation.")
    dates = z["dates"].astype(str)

    _, centers_latlon = build_segment_centers_latlon(
        segments_csv=args.segments_csv,
        nodes_csv=args.nodes_csv,
        segment_index_csv=args.segment_index_csv
    )

    plot_wpos_vs_distance(
        values=values,
        dates=dates,
        time_of_day=time_of_day,
        centers_latlon=centers_latlon,
        tau_max=args.tau_max,
        days=args.days,
        start_hhmm=args.start_hhmm,
        duration_hours=args.duration_hours,
        num_pairs=args.num_pairs,
        max_x_km=args.max_x_km,
        seed=args.seed,
        shuffle_bin_minutes=args.shuffle_bin_minutes,
        require_full_coverage=(not args.no_full_coverage),
        fixed_end_date=args.fixed_end_date,
        Wmin_line=args.Wmin_line,
        Dmin_line_km=args.Dmin_line_km,
        Dmax_line_km=args.Dmax_line_km,
        out_png=args.out_png,
    )


if __name__ == "__main__":
    main()
