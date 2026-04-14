import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def get_narx_features(df, na, nb):
    features = []

    # Output lags
    for lag in range(1, na + 1):
        features += [f"y1_lag_{lag}", f"y2_lag_{lag}"]

    # Input lags
    for lag in range(0, nb + 1):
        features += [f"u1_lag_{lag}", f"u2_lag_{lag}"]

    return features


def optimize_na_nb(df, na_range, nb_range, n_splits=5):

    target_cols = ["y1_target", "y2_target"]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []

    for na in na_range:
        for nb in nb_range:
            features = get_narx_features(df, na, nb)

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
                fold_errors.append(mse)

            avg_error = np.mean(fold_errors)

            results.append({"na": na, "nb": nb, "MSE": avg_error})

            print(f"na={na}, nb={nb}, MSE={avg_error:.4f}")

    results_df = pd.DataFrame(results)

    return results_df


def plot_na_nb_results(results_df):
    pivot = results_df.pivot(index="na", columns="nb", values="MSE")

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot, origin="lower", aspect="auto")
    plt.colorbar(label="MSE")

    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    plt.xlabel("nb (input lag)")
    plt.ylabel("na (output lag)")
    plt.title("CV Error Heatmap (na vs nb)")

    plt.show()


df = pd.read_pickle("../../data/interim/engineeredfeatures_df.pkl")

na_range = range(0, 10)
nb_range = range(0, 6)

results_df = optimize_na_nb(df, na_range, nb_range)
plot_na_nb_results(results_df)

# from gridsearch

best_na = 1
best_nb = 1


def build_final_narx_df(df, na, nb, target_cols=["y1_target", "y2_target"]):
    """
    Build final dataframe based on optimal NARX structure.

    Parameters:
        df (pd.DataFrame): original dataframe with all lag features
        na (int): output lag
        nb (int): input lag
        target_cols (list): target columns

    Returns:
        df_final (pd.DataFrame)
        feature_cols (list)
    """

    feature_cols = []

    # --- Output lags ---
    for lag in range(0, na + 1):
        feature_cols += [f"y1_lag_{lag}", f"y2_lag_{lag}"]

    # --- Input lags ---
    for lag in range(0, nb + 1):
        feature_cols += [f"u1_lag_{lag}", f"u2_lag_{lag}"]

    # Keep only existing columns (safety)
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Final dataframe
    df_final = df[feature_cols + target_cols].copy()

    return df_final, feature_cols


df_final, features = build_final_narx_df(df, best_na, best_nb)

print("Number of features:", len(features))
df_final.head()

df_final = df_final.dropna().reset_index(drop=True)

df_final.to_pickle("../../data/processed/best_features_df.pkl")
