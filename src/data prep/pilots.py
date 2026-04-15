import numpy as np
import pandas as pd


def simulate_narx1(n_steps=1000, noise_std=(0.1, 0.1), max_lag=2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Initialize
    y1 = np.zeros(n_steps + 1)
    y2 = np.zeros(n_steps + 1)
    u = np.random.uniform(-2, 2, size=n_steps + 1)

    w1 = np.random.normal(0, noise_std[0], size=n_steps + 1)
    w2 = np.random.normal(0, noise_std[1], size=n_steps + 1)

    # Simulate system
    for k in range(1, n_steps):
        y1[k + 1] = 0.5 * y2[k - 1] + np.sin(y2[k]) + 0.3 * u[k - 1] + w1[k + 1]

        y2[k + 1] = 0.5 * y1[k - 1] + np.sin(y1[k]) + 0.2 * u[k] + w2[k + 1]

    # Base DataFrame
    df = pd.DataFrame({"y1": y1[:n_steps], "y2": y2[:n_steps], "u": u[:n_steps]})

    # Targets: y(k+1)
    df["y1_target"] = df["y1"].shift(-1)
    df["y2_target"] = df["y2"].shift(-1)

    # Lagged features (lag_0 = current k)
    for lag in range(max_lag + 1):
        df[f"y1_lag_{lag}"] = df["y1"].shift(lag)
        df[f"y2_lag_{lag}"] = df["y2"].shift(lag)
        df[f"u_lag_{lag}"] = df["u"].shift(lag)

    # Drop rows with NaNs (from lagging and forward shift)
    df = df.dropna().reset_index(drop=True)

    return df


df = simulate_narx1(n_steps=2000, noise_std=(0.05, 0.05), max_lag=10, seed=42)
print(df.head())


df.to_pickle("../../data/simulated/pilot1_df.pkl")


def simulate_narx2(n_steps=1000, noise_std=(0.05, 0.05), max_lag=2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Initialize signals
    y1 = np.zeros(n_steps + 1)
    y2 = np.zeros(n_steps + 1)

    # Two inputs now
    u1 = np.random.uniform(-1, 1, size=n_steps + 1)
    u2 = np.random.uniform(-1, 1, size=n_steps + 1)

    # Noise
    w1 = np.random.normal(0, noise_std[0], size=n_steps + 1)
    w2 = np.random.normal(0, noise_std[1], size=n_steps + 1)

    # Simulation (start at k=2 because we need k-2)
    for k in range(2, n_steps):
        denom1 = 1 + y2[k - 1] ** 2 + y2[k - 2] ** 2
        denom2 = 1 + y1[k - 1] ** 2 + y1[k - 2] ** 2

        y1[k + 1] = (
            y1[k] * y1[k - 1] * y1[k - 2] * (y1[k - 2] - 1) * u2[k - 1] + u2[k]
        ) / denom1 + w1[k + 1]

        y2[k + 1] = (
            y2[k] * y2[k - 1] * y2[k - 2] * (y2[k - 2] - 1) * u1[k - 1] + u1[k]
        ) / denom2 + w2[k + 1]

    # Base DataFrame
    df = pd.DataFrame(
        {
            "y1": y1[:n_steps],
            "y2": y2[:n_steps],
            "u1": u1[:n_steps],
            "u2": u2[:n_steps],
        }
    )

    # Targets
    df["y1_target"] = df["y1"].shift(-1)
    df["y2_target"] = df["y2"].shift(-1)

    # Lagged features
    for lag in range(max_lag + 1):
        df[f"y1_lag_{lag}"] = df["y1"].shift(lag)
        df[f"y2_lag_{lag}"] = df["y2"].shift(lag)
        df[f"u1_lag_{lag}"] = df["u1"].shift(lag)
        df[f"u2_lag_{lag}"] = df["u2"].shift(lag)

    # Clean NaNs
    df = df.dropna().reset_index(drop=True)

    return df


df = simulate_narx2(n_steps=2000, noise_std=(0.05, 0.05), max_lag=3, seed=42)
print(df.head())

df.to_pickle("../../data/simulated/pilot2_df.pkl")
