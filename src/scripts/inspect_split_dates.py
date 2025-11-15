import numpy as np
import pandas as pd

mkt = pd.read_csv("/home/zxy/hybrid_advisor_offline/data/mkt_data.csv", index_col="Date", parse_dates=True)

def inspect(meta_path, name):
    meta = np.load(meta_path, allow_pickle=True)
    starts = meta["episode_start_idx"].astype(int)
    print(f"\n[{name}] episodes={len(starts)}")
    print("  earliest start_idx:", starts.min(), "date:", mkt.index[starts.min()])
    print("  latest   start_idx:", starts.max(), "date:", mkt.index[starts.max()])

print("Market data range:", mkt.index.min(), "->", mkt.index.max())
inspect("/home/zxy/hybrid_advisor_offline/data/offline_dataset_train_behavior.npz", "train")
inspect("/home/zxy/hybrid_advisor_offline/data/offline_dataset_val_behavior.npz", "val")
