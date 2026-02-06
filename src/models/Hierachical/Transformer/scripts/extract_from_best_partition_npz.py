import argparse
import numpy as np
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="best_partition.npz")
    ap.add_argument("--out_dir", type=str, default="", help="where to save .npy (default = same folder)")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    labels = data["labels_hard"].astype("int64")
    Tran = data["Tran_soft"].astype("float32")
    A_R = data["A_R"].astype("float32")

    out_dir = args.out_dir if args.out_dir else os.path.dirname(args.npz)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "region_id_of_node.npy"), labels)
    np.save(os.path.join(out_dir, "Tran.npy"), Tran)
    np.save(os.path.join(out_dir, "A_R.npy"), A_R)

    print("Saved to", out_dir)
    print("labels", labels.shape, "Tran", Tran.shape, "A_R", A_R.shape)

if __name__ == "__main__":
    main()
