import numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=str, required=True)
    ap.add_argument("--tran", type=str, required=True)
    ap.add_argument("--ar", type=str, required=True)
    args = ap.parse_args()

    labels = np.load(args.labels)
    Tran = np.load(args.tran)
    A_R = np.load(args.ar)

    N = labels.shape[0]
    R = Tran.shape[1]
    print("N", N, "R", R)
    print("labels min/max", labels.min(), labels.max(), "unique", len(np.unique(labels)))
    row_sum = Tran.sum(axis=1)
    print("Tran row sum: mean", row_sum.mean(), "min", row_sum.min(), "max", row_sum.max())
    print("A_R shape", A_R.shape, "nonzero", np.count_nonzero(A_R))

if __name__ == "__main__":
    main()
