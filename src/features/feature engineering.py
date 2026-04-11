import pandas as pd

df = pd.read_pickle("../../data/interim/training_df.pkl")


def engineer_features(df, max_lag=10):
    """
    Create lagged features and rolling statistics.

    Parameters:
        df (pd.DataFrame): must contain ['u1','u2','y1','y2']
        max_lag (int): maximum lag AND rolling window size

    Returns:
        pd.DataFrame
    """

    df_feat = df.copy()
    variables = ["u1", "u2", "y1", "y2"]

    # --- Lag features ---
    for var in variables:
        for lag in range(1, max_lag + 1):
            df_feat[f"{var}_lag_{lag}"] = df_feat[var].shift(lag)

    # --- Rolling statistics (window = max_lag) ---
    for var in variables:
        df_feat[f"{var}_roll_mean"] = df_feat[var].rolling(window=max_lag).mean()
        df_feat[f"{var}_roll_std"] = df_feat[var].rolling(window=max_lag).std()
        df_feat[f"{var}_roll_min"] = df_feat[var].rolling(window=max_lag).min()
        df_feat[f"{var}_roll_max"] = df_feat[var].rolling(window=max_lag).max()

    # --- Targets ---
    df_feat["y1_target"] = df_feat["y1"].shift(-1)
    df_feat["y2_target"] = df_feat["y2"].shift(-1)

    df_feat = df_feat.dropna()

    return df_feat


df_features = engineer_features(df, max_lag=60)

df_features.to_pickle("../../data/interim/engineeredfeatures_df.pkl")
