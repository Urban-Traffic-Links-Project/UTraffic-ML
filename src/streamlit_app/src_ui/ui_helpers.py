# src_ui/ui_helpers.py
import pandas as pd

def summarize_df(df: pd.DataFrame) -> dict:
    info = {
        "rows": len(df),
        "cols": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": float(df.isna().sum().sum() / (df.shape[0] * df.shape[1] + 1e-9)),
        "n_segments": int(df["segment_id"].nunique()) if "segment_id" in df.columns else None,
    }
    return info

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rep = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().sum() / len(df)).round(4)
    }).sort_values("missing_count", ascending=False)
    rep = rep[rep["missing_count"] > 0]
    return rep
