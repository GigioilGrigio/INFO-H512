import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tqdm.auto import tqdm


def make_narx_supervised(df, target_cols, lag, feature_cols=None):
    """
    Build supervised NARX data from a raw time-series dataframe.

    X_t = [y_{t-1}, ..., y_{t-lag}, exog_t]
    y_t = current target vector at time t
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in target_cols]

    if len(df) <= lag:
        raise ValueError(f"Dataframe too short for lag={lag}.")

    X, y = [], []

    for t in tqdm(range(lag, len(df)), desc="Building supervised NARX data"):
        row_features = []

        for k in range(1, lag + 1):
            row_features.extend(df[target_cols].iloc[t - k].to_numpy())

        row_features.extend(df[feature_cols].iloc[t].to_numpy())

        X.append(row_features)
        y.append(df[target_cols].iloc[t].to_numpy())

    return np.asarray(X), np.asarray(y), feature_cols


def evaluate_mlp_narx_long_term(
    df,
    test_size=0.2,
    lag=5,
    hidden_layer_sizes=(32, 16),
    random_state=42,
):
    """
    Train and evaluate a simple NARX neural network with recursive long-term prediction.
    """

    target_cols = ["y1_target", "y2_target"]
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    feature_cols = [c for c in df.columns if c not in target_cols]

    if len(train_df) <= lag:
        raise ValueError(f"Train split too short for lag={lag}.")

    # --- Build supervised training set ---
    X_train, y_train, feature_cols = make_narx_supervised(
        train_df, target_cols=target_cols, lag=lag, feature_cols=feature_cols
    )

    # --- Model pipeline ---
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    max_iter=3000,
                    early_stopping=True,
                    n_iter_no_change=30,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # --- Train ---
    model.fit(X_train, y_train)

    # --- Recursive long-term prediction on test set ---
    history = list(train_df[target_cols].iloc[-lag:].to_numpy())

    preds = []
    for i in tqdm(range(len(test_df)), desc="Recursive prediction"):
        exog_t = test_df.iloc[i][feature_cols].to_numpy()

        lag_block = np.concatenate([history[-k] for k in range(1, lag + 1)])
        X_t = np.concatenate([lag_block, exog_t]).reshape(1, -1)

        y_pred_t = model.predict(X_t)[0]
        preds.append(y_pred_t)

        history.append(y_pred_t)
        if len(history) > lag:
            history.pop(0)

    y_pred = np.asarray(preds)
    y_test = test_df[target_cols].to_numpy()

    # --- Metrics ---
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    nmse = mse / np.var(y_test)

    print(f"Lag: {lag}, RMSE: {rmse:.4f}, NMSE: {nmse:.4f}, Hidden Layers: {hidden_layer_sizes}")

    # --- Plot y1 ---
    plt.figure(figsize=(10, 3))
    plt.plot(y_test[:, 0], label="y1 true")
    plt.plot(y_pred[:, 0], label="y1 pred")
    plt.title(f"MLP NARX Long-Term - y1 (lag={lag})")
    plt.legend()
    plt.grid()

    # --- Plot y2 ---
    plt.figure(figsize=(10, 3))
    plt.plot(y_test[:, 1], label="y2 true")
    plt.plot(y_pred[:, 1], label="y2 pred")
    plt.title(f"MLP NARX Long-Term - y2 (lag={lag})")
    plt.legend()
    plt.grid()

    plt.show()

    results = pd.DataFrame(
        [{
            "Model": f"MLP_NARX_L{lag}",
            "RMSE": rmse,
            "NMSE": nmse,
            "Hidden Layers": hidden_layer_sizes
        }]
    )

    return results, model, y_pred, y_test


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


# --- Example usage ---
df = pd.read_pickle("../../data/processed/best_features_df.pkl")

# evaluate narx mlp with variable lag:
# for lag in [1, 2,3,4,5,6,7,8,9,10]:
#   results, model, y_pred, y_test = evaluate_mlp_narx_long_term(
#     df,
#     test_size=0.2,
#     lag=lag,  # change this freely
#     hidden_layer_sizes=(32, 16),
#   )

# evaluate with a fixed lag of 3 but variable hidden layer sizes:
# for hidden_sizes in [(5,5), (32, 16), (64, 32), (128, 64), (256, 128), (512, 256), (1024, 512), (2048, 1024), (4096, 2048)]:
#   results, model, y_pred, y_test = evaluate_mlp_narx_long_term(
#     df,
#     test_size=0.2,
#     lag=3,  # fixed lag
#     hidden_layer_sizes=hidden_sizes,  # variable hidden layer sizes
#   )

results, model, y_pred, y_test = evaluate_mlp_narx_long_term(
  df,
  test_size=0.2,
  lag=3,  # change this freely
  hidden_layer_sizes=(4096, 2048),
)