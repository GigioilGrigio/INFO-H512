import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tqdm.auto import tqdm


def make_narx_supervised(df, target_cols, lag, feature_cols=None):
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
    n_epochs=200,
):
    target_cols = ["y1_target", "y2_target"]
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    feature_cols = [c for c in df.columns if c not in target_cols]

    if len(train_df) <= lag:
        raise ValueError(f"Train split too short for lag={lag}.")

    t0 = time.perf_counter()
    X_train, y_train, feature_cols = make_narx_supervised(
        train_df, target_cols=target_cols, lag=lag, feature_cols=feature_cols
    )
    print(f"Built training data in {time.perf_counter() - t0:.2f}s")

    # Scale manually so we can show progress during training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Use partial_fit so we can show a tqdm bar over epochs
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        random_state=random_state,
        max_iter=1,          # one epoch per partial_fit call
        warm_start=True,
        shuffle=True,
    )

    print("Training model...")
    train_start = time.perf_counter()
    for epoch in tqdm(range(n_epochs), desc="Training MLP", leave=True):
        model.partial_fit(X_train_scaled, y_train)
        # optional live loss display
        tqdm.write(f"Epoch {epoch + 1}/{n_epochs} | loss = {model.loss_:.6f}") if (epoch + 1) % 25 == 0 else None

    print(f"Training finished in {time.perf_counter() - train_start:.2f}s")

    # Recursive long-term prediction
    history = list(train_df[target_cols].iloc[-lag:].to_numpy())

    preds = []
    pred_start = time.perf_counter()
    for i in tqdm(range(len(test_df)), desc="Recursive prediction"):
        exog_t = test_df.iloc[i][feature_cols].to_numpy()

        lag_block = np.concatenate([history[-k] for k in range(1, lag + 1)])
        X_t = np.concatenate([lag_block, exog_t]).reshape(1, -1)
        X_t_scaled = scaler.transform(X_t)

        y_pred_t = model.predict(X_t_scaled)[0]
        preds.append(y_pred_t)

        history.append(y_pred_t)
        if len(history) > lag:
            history.pop(0)

    print(f"Prediction finished in {time.perf_counter() - pred_start:.2f}s")

    y_pred = np.asarray(preds)
    y_test = test_df[target_cols].to_numpy()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    nmse = mse / np.var(y_test)

    print(f"Lag: {lag}, RMSE: {rmse:.4f}, NMSE: {nmse:.4f}, Hidden Layers: {hidden_layer_sizes}")

    plt.figure(figsize=(10, 3))
    plt.plot(y_test[:, 0], label="y1 true")
    plt.plot(y_pred[:, 0], label="y1 pred")
    plt.title(f"MLP NARX Long-Term - y1 (lag={lag})")
    plt.legend()
    plt.grid()

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

    return results, model, y_pred, y_test, scaler


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

df = pd.read_pickle("../../data/processed/best_features_df.pkl")

results, model, y_pred, y_test, scaler = evaluate_mlp_narx_long_term(
    df,
    test_size=0.2,
    lag=3,
    hidden_layer_sizes=(4096, 2048),
    n_epochs=200,
)