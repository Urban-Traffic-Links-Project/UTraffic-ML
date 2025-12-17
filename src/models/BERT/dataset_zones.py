# src/dataset_zones.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from matrix_correla import (
    load_traffic_tensor,
    load_graph_topology,
    ZoneBuilder, DBSCANParams, ZoneParams, CorrParams
)

# -------------------------
# Session helper
# -------------------------

def build_session_id(dates: np.ndarray, time_of_day: np.ndarray,
                     morning_end: float = 10/24, afternoon_start: float = 15/24) -> np.ndarray:
    """
    Create session id per time index:
      - morning session: [7..10) -> time_of_day in [7/24, 10/24)
      - afternoon session: [15..18) -> time_of_day in [15/24, 18/24)
    Your preprocessor already filtered data to these ranges,
    but we still mark sessions to prevent crossing.
    """
    # time_of_day is [0,1]
    # session label: 0 = morning, 1 = afternoon, -1 = other
    sid = np.full((len(dates),), -1, dtype=np.int32)
    sid[(time_of_day < morning_end)] = 0
    sid[(time_of_day >= afternoon_start)] = 1
    # session group id includes date to prevent crossing day boundary
    # map date string to int
    # Use stable hashing by unique sorted dates
    unique_dates = {d: i for i, d in enumerate(sorted(set(dates.tolist())))}
    date_id = np.array([unique_dates[d] for d in dates.tolist()], dtype=np.int32)
    session_gid = date_id * 10 + sid  # unique per (date, session)
    return session_gid


def chronological_day_split(dates: np.ndarray,
                            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split by unique dates in chronological order.
    Return sets of date strings: train/val/test.
    """
    uniq = sorted(set(dates.tolist()))
    n = len(uniq)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    train_dates = set(uniq[:n_train])
    val_dates = set(uniq[n_train:n_train+n_val])
    test_dates = set(uniq[n_train+n_val:])
    return train_dates, val_dates, test_dates


@dataclass
class DatasetParams:
    L: int = 12              # history length (time steps)
    delta: int = 1           # predict at t+delta
    hist_len_corr: int = 48  # history for correlation links (can be >= L)
    # session boundaries in normalized tod (0..1). keep consistent with preprocessor filtering.
    morning_end: float = 10/24
    afternoon_start: float = 15/24


class TrafficZoneDataset(Dataset):
    """
    Each item:
      - build zone at time t (ZoneBuilder)
      - create x_window: (L, N_zone, d_in)  (here d_in=1 with zspeed)
      - tod: (L,)
      - dow: (L,)
      - lap: (N_zone, d_spa)
      - y: (N_zone,) at t+delta
      - node_mask: (N_zone,) all ones
      - target_mask: (N_zone,) from ZoneBuilder
    """
    def __init__(
        self,
        traffic_npz: str,
        segments_csv: str,
        nodes_csv: str,
        edges_csv: str,
        segment_index_csv: str,
        split: str,  # "train" | "val" | "test"
        ds: DatasetParams = DatasetParams(),
        dbscan: DBSCANParams = DBSCANParams(),
        zone: ZoneParams = ZoneParams(),
        seed: int = 13,
    ):
        super().__init__()
        self.ds = ds
        self.split = split

        # Load tensor + topology
        self.traffic = load_traffic_tensor(traffic_npz)
        self.topo = load_graph_topology(segments_csv, nodes_csv, edges_csv, segment_index_csv)

        assert self.traffic.dates is not None, "traffic_tensor.npz must contain 'dates' for day split"
        assert self.traffic.time_of_day is not None, "traffic_tensor.npz must contain 'time_of_day'"
        assert self.traffic.day_of_week is not None, "traffic_tensor.npz must contain 'day_of_week'"

        # Split by day (chronological)
        tr_dates, va_dates, te_dates = chronological_day_split(self.traffic.dates)
        if split == "train":
            date_set = tr_dates
        elif split == "val":
            date_set = va_dates
        elif split == "test":
            date_set = te_dates
        else:
            raise ValueError("split must be train|val|test")

        # Session id (prevents crossing 10h->15h etc.)
        self.session_gid = build_session_id(
            dates=self.traffic.dates,
            time_of_day=self.traffic.time_of_day,
            morning_end=ds.morning_end,
            afternoon_start=ds.afternoon_start
        )

        # Valid t list
        T = self.traffic.values.shape[0]
        L = ds.L
        delta = ds.delta

        valid_t: List[int] = []
        for t in range(T):
            # must have label at t+delta
            if t + delta >= T:
                continue
            # day must be in split
            if self.traffic.dates[t] not in date_set:
                continue
            # window must be fully inside same session_gid
            t0 = t - L + 1
            if t0 < 0:
                continue
            sid = self.session_gid[t]
            if sid < 0:
                continue
            if not np.all(self.session_gid[t0:t+1] == sid):
                continue
            # also require label time is same split day (avoid leakage across day boundary)
            if self.traffic.dates[t + delta] != self.traffic.dates[t]:
                continue
            # ensure label inside same session too (optional but consistent)
            if self.session_gid[t + delta] != sid:
                continue

            valid_t.append(t)

        self.valid_t = np.array(valid_t, dtype=np.int64)

        # ZoneBuilder (randomness inside sampling seed cluster)
        self.zone_builder = ZoneBuilder(
            traffic=self.traffic,
            topo=self.topo,
            dbscan=dbscan,
            zone=zone,
            random_seed=seed,
        )

        # feature config
        self.d_in = 1  # only zspeed for now

    def __len__(self):
        return int(self.valid_t.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        t = int(self.valid_t[idx])
        L = int(self.ds.L)
        delta = int(self.ds.delta)

        # Build zone at time t
        z = self.zone_builder.build_zone(
            t=t,
            hist_len=int(self.ds.hist_len_corr),
            return_lap=True
        )
        zone_indices = np.array(z["zone_indices"], dtype=np.int64)   # (N_zone,)
        target_mask = z["target_mask_zone"].astype(np.int8)          # (N_zone,)
        lap = z["lap_eigvec"].astype(np.float32) if z["lap_eigvec"] is not None else np.zeros((len(zone_indices), 0), np.float32)

        # Build window [t-L+1 .. t]
        t0 = t - L + 1
        x = self.traffic.values[t0:t+1, :][:, zone_indices]  # (L, N_zone)
        x = x[..., None].astype(np.float32)                  # (L, N_zone, 1)

        tod = self.traffic.time_of_day[t0:t+1].astype(np.float32)    # (L,)
        dow = self.traffic.day_of_week[t0:t+1].astype(np.int64)      # (L,)

        # label at t+delta
        y = self.traffic.is_congested[t + delta, :][zone_indices].astype(np.float32)  # (N_zone,)

        node_mask = np.ones((zone_indices.shape[0],), dtype=np.int8)

        return {
            "x": torch.from_numpy(x),                  # (L,Nz,1)
            "tod": torch.from_numpy(tod),              # (L,)
            "dow": torch.from_numpy(dow),              # (L,)
            "lap": torch.from_numpy(lap),              # (Nz,d_spa)
            "y": torch.from_numpy(y),                  # (Nz,)
            "node_mask": torch.from_numpy(node_mask),  # (Nz,)
            "target_mask": torch.from_numpy(target_mask),  # (Nz,)
            "meta": z["meta"],                         # dict (debug)
        }
