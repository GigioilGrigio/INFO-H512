import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def evaluate_rf_narx_long_term(df, test_size=0.2):
    """
    Train and evaluate Random Forest with recursive long-term prediction.

    Same structure as evaluate_models_time_series, but:
    - Uses recursive forecasting
    """

    # --- Targets & features ---
    target_cols = ["y1_target", "y2_target"]
    feature_cols = [c for c in df.columns if c not in target_cols]

    # --- Time-based split ---
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_cols].values
    y_test = test_df[target_cols].values

    # --- Model ---
    model = RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    )

    # --- Train ---
    model.fit(X_train, y_train)

    # --- Recursive prediction ---
    preds = []
    current_row = test_df.iloc[0].copy()

    for i in range(len(test_df)):
        X = current_row[feature_cols].values.reshape(1, -1)
        y_pred = model.predict(X)[0]
        preds.append(y_pred)

        if i < len(test_df) - 1:
            next_row = test_df.iloc[i + 1].copy()

            # update lagged outputs (assumes lag_1 exists)
            if "y1_lag_1" in next_row:
                next_row["y1_lag_1"] = y_pred[0]
            if "y2_lag_1" in next_row:
                next_row["y2_lag_1"] = y_pred[1]

            current_row = next_row

    y_pred = np.array(preds)

    # --- Metrics (same style as your first function) ---
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    nmse = mse / np.var(y_test)

    print("=== Random Forest (Long-Term) ===")
    print(f"RMSE: {rmse:.4f}, NMSE: {nmse:.4f}")

    # --- Plot y1 ---
    plt.figure(figsize=(10, 3))
    plt.plot(y_test[:, 0], label="y1 true")
    plt.plot(y_pred[:, 0], label="y1 pred")
    plt.title("RandomForest Long-Term - y1")
    plt.legend()
    plt.grid()

    # --- Plot y2 ---
    plt.figure(figsize=(10, 3))
    plt.plot(y_test[:, 1], label="y2 true")
    plt.plot(y_pred[:, 1], label="y2 pred")
    plt.title("RandomForest Long-Term - y2")
    plt.legend()
    plt.grid()

    plt.show()

    # --- Return results in same format ---
    results = pd.DataFrame(
        [{"Model": "RandomForest_LongTerm", "RMSE": rmse, "NMSE": nmse}]
    )

    return results, model, y_pred, y_test


df = pd.read_pickle("../../data/processed/best_features_df.pkl")
results, model, y_pred, y_test = evaluate_rf_narx_long_term(df, test_size=0.2)


def plot_residual_acf_pacf(y_true, y_pred, lags=20):
    residuals = y_true - y_pred

    for i, name in enumerate(["y1", "y2"]):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plot_acf(residuals[:, i], lags=lags, ax=plt.gca())
        plt.title(f"ACF Residuals - {name}")

        plt.subplot(1, 2, 2)
        plot_pacf(residuals[:, i], lags=lags, ax=plt.gca(), method="ywm")
        plt.title(f"PACF Residuals - {name}")

        plt.tight_layout()
        plt.show()


plot_residual_acf_pacf(y_test, y_pred, lags=40)
