import pandas as pd
import numpy as np

data = np.load("../../data/raw/StudentdataNARX.npz")

Utr = data["Utr"]
Ytr = data["Ytr"]
Uts1 = data["Uts1"]
Uts2 = data["Uts2"]
# Create timestamp (adjust frequency as needed)
timestamp = pd.date_range(start="2026-01-01", periods=Utr.shape[0], freq="s")

# Build DataFrame
df = pd.DataFrame(
    {
        "timestamp": timestamp,
        "u1": Utr[:, 0],
        "u2": Utr[:, 1],
        "y1": Ytr[:, 0],
        "y2": Ytr[:, 1],
    }
)

df.set_index("timestamp", inplace=True)

df.to_pickle("../../data/interim/training_df.pkl")
