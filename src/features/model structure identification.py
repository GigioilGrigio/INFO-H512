import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


def build_structured_interactions(df, na=3, nb=3):

    feature_cols = []

    # ----------------------------
    # 1. RAW FEATURES (just select)
    # ----------------------------
    y1_lags = [f"y1_lag_{i}" for i in range(0, na + 1)]
    y2_lags = [f"y2_lag_{i}" for i in range(0, na + 1)]

    u1_lags = [f"u1_lag_{i}" for i in range(0, nb + 1)]
    u2_lags = [f"u2_lag_{i}" for i in range(0, nb + 1)]

    feature_cols += y1_lags + y2_lags + u1_lags + u2_lags

    # ----------------------------
    # 2. OUTPUT - OUTPUT INTERACTIONS
    # (only cross-output)
    # ----------------------------
    for i in range(0, na + 1):
        for j in range(0, na + 1):
            col_1 = f"y1_lag_{i}"
            col_2 = f"y2_lag_{j}"

            new_col = f"y1y2_{i}_{j}"
            df[new_col] = df[col_1] * df[col_2]
            feature_cols.append(new_col)

    # ----------------------------
    # 3. OUTPUT - INPUT INTERACTIONS
    # ----------------------------

    # y1 interactions
    for i in range(0, na + 1):
        for j in range(0, nb + 1):
            for u in ["u1", "u2"]:
                df[f"y1_{i}_{u}_{j}"] = df[f"y1_lag_{i}"] * df[f"{u}_lag_{j}"]
                feature_cols.append(f"y1_{i}_{u}_{j}")

    # y2 interactions
    for i in range(0, na + 1):
        for j in range(0, nb + 1):
            for u in ["u1", "u2"]:
                df[f"y2_{i}_{u}_{j}"] = df[f"y2_lag_{i}"] * df[f"{u}_lag_{j}"]
                feature_cols.append(f"y2_{i}_{u}_{j}")

    return df, feature_cols


def get_narx_features(df, na, nb, input_cols=None):
    features = []

    # Default: assume single input "u"
    if input_cols is None:
        input_cols = ["u"]

    # Output lags
    for lag in range(0, na + 1):
        features += [f"y1_lag_{lag}", f"y2_lag_{lag}"]

    # Input lags
    for lag in range(0, nb + 1):
        for u in input_cols:
            features.append(f"{u}_lag_{lag}")

    return features


def optimize_na_nb(df, na_range, nb_range, n_splits=5):

    target_cols = ["y1_target", "y2_target"]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []

    for na in na_range:
        for nb in nb_range:
            features = get_narx_features(df, na, nb, ["u1", "u2"])

            # skip if features missing
            features = [f for f in features if f in df.columns]

            X = df[features]
            y = df[target_cols]

            fold_errors = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model = MultiOutputRegressor(
                    RandomForestRegressor(n_estimators=100, random_state=42)
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                nmse = mse / np.var(y_test)
                fold_errors.append(nmse)

            avg_error = np.mean(fold_errors)

            results.append({"na": na, "nb": nb, "NMSE": avg_error})

            print(f"na={na}, nb={nb}, NMSE={avg_error:.4f}")

    results_df = pd.DataFrame(results)

    return results_df


def optimize_na_nb_svr(df, na_range, nb_range, n_splits=5, svr_kwargs=None):

    target_cols = ["y1_target", "y2_target"]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []

    # default SVR settings (you can tune this)
    if svr_kwargs is None:
        svr_kwargs = {"kernel": "rbf", "C": 10.0, "gamma": "scale", "epsilon": 0.1}

    for na in na_range:
        for nb in nb_range:
            d, features = build_structured_interactions(df, na, nb)

            X = d[features]
            y = d[target_cols]

            fold_errors = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model = MultiOutputRegressor(SVR(**svr_kwargs))

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                nmse = mse / np.var(y_test)

                fold_errors.append(nmse)

            avg_error = np.mean(fold_errors)

            results.append({"na": na, "nb": nb, "NMSE": avg_error})

            print(f"na={na}, nb={nb}, NMSE={avg_error:.4f}")

    return pd.DataFrame(results)


def plot_na_nb_results(results_df):
    pivot = results_df.pivot(index="na", columns="nb", values="NMSE")

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot, origin="lower", aspect="auto")
    plt.colorbar(label="NMSE")

    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    plt.xlabel("nb (input lag)")
    plt.ylabel("na (output lag)")
    plt.title("CV Error Heatmap (na vs nb)")

    plt.show()


df = pd.read_pickle("../../data/interim/engineeredfeatures_df.pkl")
pilot1 = pd.read_pickle("../../data/simulated/pilot1_df.pkl")
pilot2 = pd.read_pickle("../../data/simulated/pilot2_df.pkl")

na_range = range(0, 3)
nb_range = range(0, 3)

results_df = optimize_na_nb(pilot2, na_range, nb_range)
results_df = optimize_na_nb_svr(df, na_range, nb_range)
plot_na_nb_results(results_df)

# from gridsearch

best_na = 2
best_nb = 0


def build_final_narx_df(
    df, na, nb, input_cols=None, target_cols=("y1_target", "y2_target")
):
    """
    Build final dataframe based on optimal NARX structure.

    Parameters:
        df (pd.DataFrame): original dataframe with all lag features
        na (int): output lag
        nb (int): input lag
        input_cols (list or None): list of input names (e.g., ["u"] or ["u1","u2"])
        target_cols (list/tuple): target columns

    Returns:
        df_final (pd.DataFrame)
        feature_cols (list)
    """

    # Default: single input
    if input_cols is None:
        input_cols = ["u"]

    feature_cols = []

    # --- Output lags ---
    for lag in range(0, na + 1):
        feature_cols += [f"y1_lag_{lag}", f"y2_lag_{lag}"]

    # --- Input lags ---
    for lag in range(0, nb + 1):
        for u in input_cols:
            feature_cols.append(f"{u}_lag_{lag}")

    # Keep only existing columns (safety)
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Final dataframe
    df_final = df[feature_cols + list(target_cols)].copy()

    return df_final, feature_cols


df_final, features = build_final_narx_df(pilot2, best_na, best_nb, ["u1", "u2"])

print("Number of features:", len(features))
df_final.head()

df_final = df_final.dropna().reset_index(drop=True)

df_final.to_pickle("../../data/processed/best_features_pilot2.pkl")
