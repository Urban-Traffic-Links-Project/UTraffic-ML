from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet
import argparse

from branchB_xt_graph_feature_utils import build_graph_signal_topk

# =========================
# CONFIG
# =========================
HORIZONS = list(range(1, 10))

ALPHA = 0.001
L1_RATIO = 0.5
MAX_ITER = 500

TOPK = 4
GAMMA = 0.3


# =========================
# LOAD DATA
# =========================
def load_split(common_dir, split):
    d = common_dir / split
    return {
        "G": np.load(d / "G_weight_series.npy", mmap_mode="r"),
        "z": np.load(d / "z.npy", mmap_mode="r"),
        "meta": pd.read_csv(d / "G_series_meta.csv"),
    }


def iter_pairs(meta, horizon):
    for i in range(len(meta) - horizon):
        yield i, i + horizon


# =========================
# BUILD DATASET
# =========================
def build_dataset(split_data, horizon):
    X, Y = [], []

    G_series = split_data["G"]
    z = split_data["z"]
    meta = split_data["meta"]

    for t, t2 in iter_pairs(meta, horizon):
        x_t = z[t]
        y = z[t2]

        G_used = G_series[t]

        gx = build_graph_signal_topk(G_used, x_t, k=TOPK, gamma=GAMMA)

        feat = np.concatenate([x_t, gx])

        X.append(feat)
        Y.append(y)

    return np.array(X), np.array(Y)


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.data_dir)

    train = load_split(root, "train")
    val = load_split(root, "val")
    test = load_split(root, "test")

    results = []

    for h in HORIZONS:
        print(f"\n=== Horizon {h} ===")

        X_train, Y_train = build_dataset(train, h)

        model = MultiTaskElasticNet(
            alpha=ALPHA,
            l1_ratio=L1_RATIO,
            max_iter=MAX_ITER,
        )

        model.fit(X_train, Y_train)

        for name, split in [("val", val), ("test", test)]:
            X, Y = build_dataset(split, h)

            pred = model.predict(X)

            mae = np.mean(np.abs(pred - Y))
            rmse = np.sqrt(np.mean((pred - Y) ** 2))

            print(name, mae, rmse)

            results.append({
                "method": f"topk_{TOPK}",
                "split": name,
                "lag": h,
                "mae": mae,
                "rmse": rmse,
            })

    df = pd.DataFrame(results)

    out = Path("results/topk_xt")
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "metrics.csv", index=False)

    print("Saved results!")


if __name__ == "__main__":
    main()