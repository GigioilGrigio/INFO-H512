import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


df = pd.read_pickle("../../data/processed/best_features_df.pkl")


def evaluate_models_time_series(df, test_size=0.2):
    """
    Train and compare multiple models on time series data.

    Assumes df already contains ONLY:
    - feature columns
    - y1_target, y2_target

    Returns:
        pd.DataFrame with results
    """

    # --- Automatically detect targets ---
    target_cols = ["y1_target", "y2_target"]

    # everything else = features
    feature_cols = [c for c in df.columns if c not in target_cols]

    # --- Time-based split (NO SHUFFLE) ---
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df[target_cols]
    y_test = test_df[target_cols]

    # --- Models ---
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42)
        ),
    }

    results = []

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # --- Metrics ---
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        nmse = mse / np.var(y_test.values)

        results.append({"Model": name, "RMSE": rmse, "NMSE": nmse})

        # --- Plot y1 ---
        plt.figure(figsize=(10, 3))
        plt.plot(y_test.values[:, 0], label="y1 true")
        plt.plot(y_pred[:, 0], label="y1 pred")
        plt.title(f"{name} - y1")
        plt.legend()
        plt.grid()

        # --- Plot y2 ---
        plt.figure(figsize=(10, 3))
        plt.plot(y_test.values[:, 1], label="y2 true")
        plt.plot(y_pred[:, 1], label="y2 pred")
        plt.title(f"{name} - y2")
        plt.legend()
        plt.grid()

    plt.show()

    return pd.DataFrame(results)


results = evaluate_models_time_series(df)
print(results)


def residual_diagnostics(df, test_size=0.2, model_type="ridge", max_lag=30):
    """
    Train model, compute residuals, and plot ACF, PACF, and CCF.

    Parameters:
        df (pd.DataFrame): must contain features + ['y1_target','y2_target']
        test_size (float): proportion of test set
        model_type (str): 'ridge' or 'rf'
        max_lag (int): max lag for ACF/PACF/CCF
    """

    # --- Split ---
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    target_cols = ["y1_target", "y2_target"]
    feature_cols = [c for c in df.columns if c not in target_cols]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_cols]
    y_test = test_df[target_cols]

    # --- Model selection ---
    if model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "rf":
        model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42)
        )
    else:
        raise ValueError("model_type must be 'ridge' or 'rf'")

    # --- Train ---
    model.fit(X_train, y_train)

    # --- Predict ---
    y_pred = model.predict(X_test)

    # --- Residuals ---
    residuals = y_test.values - y_pred

    # Separate residuals
    e1 = residuals[:, 0]
    e2 = residuals[:, 1]

    # --- ACF & PACF ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    plot_acf(e1, lags=max_lag, ax=axes[0, 0])
    axes[0, 0].set_title("ACF Residuals y1")

    plot_pacf(e1, lags=max_lag, ax=axes[0, 1])
    axes[0, 1].set_title("PACF Residuals y1")

    plot_acf(e2, lags=max_lag, ax=axes[1, 0])
    axes[1, 0].set_title("ACF Residuals y2")

    plot_pacf(e2, lags=max_lag, ax=axes[1, 1])
    axes[1, 1].set_title("PACF Residuals y2")

    plt.tight_layout()

    # --- CCF function ---
    def cross_corr(x, y, max_lag):
        lags = range(0, max_lag)
        corr = [np.corrcoef(x[lag:], y[:-lag] if lag != 0 else y)[0, 1] for lag in lags]
        return lags, corr

    # --- CCF plots ---
    for i, u in enumerate(["u1", "u2"]):
        u_series = test_df[u].values

        lags, ccf_y1 = cross_corr(e1, u_series, max_lag)
        lags, ccf_y2 = cross_corr(e2, u_series, max_lag)

        plt.figure(figsize=(10, 3))
        plt.stem(lags, ccf_y1)
        plt.title(f"CCF Residual y1 vs {u}")
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.grid()

        plt.figure(figsize=(10, 3))
        plt.stem(lags, ccf_y2)
        plt.title(f"CCF Residual y2 vs {u}")
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.grid()

    plt.show()


residual_diagnostics(df, model_type="rf", max_lag=30)
