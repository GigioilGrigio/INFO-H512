import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Mean Squared Error over both outputs.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    num = np.sum((y_true - y_pred) ** 2)
    y_bar = np.mean(y_true, axis=0, keepdims=True)
    den = np.sum((y_true - y_bar) ** 2)

    if den == 0:
        return np.inf
    return num / den


def temporal_split(U: np.ndarray, Y: np.ndarray, train_ratio: float = 0.8):
    """
    Chronological split for time-series validation.
    """
    split_idx = int(len(U) * train_ratio)
    return U[:split_idx], Y[:split_idx], U[split_idx:], Y[split_idx:]


def build_narx_supervised(U: np.ndarray, Y: np.ndarray, d: int, na: int, nb: int):
    """
    Build one-step supervised data for NARX.

    Predict Y[k+1] from:
      Y[k-d], Y[k-d-1], ..., Y[k-d-na]
      U[k],   U[k-1],   ..., U[k-nb]

    Parameters
    ----------
    U : ndarray of shape (T, 2)
        Input sequence.
    Y : ndarray of shape (T, 2)
        Output sequence.
    d : int
        Output delay.
    na : int
        Number of output lags.
    nb : int
        Number of input lags.

    Returns
    -------
    X : ndarray of shape (N, n_features)
    y : ndarray of shape (N, 2)
    """
    T = len(U)
    assert U.shape[1] == 2 and Y.shape[1] == 2

    start_k = max(d + na, nb)
    end_k = T - 1  # target is Y[k+1]

    X_list = []
    y_list = []

    for k in range(start_k, end_k):
        feats = []

        # Output regressors: Y[k-d], ..., Y[k-d-na]
        for lag in range(na + 1):
            feats.extend(Y[k - d - lag])

        # Input regressors: U[k], ..., U[k-nb]
        for lag in range(nb + 1):
            feats.extend(U[k - lag])

        X_list.append(feats)
        y_list.append(Y[k + 1])

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    return X, y


def rollout_predict(model, U: np.ndarray, d: int, na: int, nb: int) -> np.ndarray:
    """
    Recursive long-term prediction using only the input sequence U.

    Initial unknown output history is filled with zeros.
    """
    T = len(U)
    Yhat = np.zeros((T, 2), dtype=float)

    def get_y(t: int):
        if t < 0:
            return np.zeros(2, dtype=float)
        return Yhat[t]

    def get_u(t: int):
        if t < 0:
            return np.zeros(2, dtype=float)
        return U[t]

    for t in range(T):
        k = t - 1
        feats = []

        # Y[k-d], ..., Y[k-d-na]
        for lag in range(na + 1):
            feats.extend(get_y(k - d - lag))

        # U[k], ..., U[k-nb]
        for lag in range(nb + 1):
            feats.extend(get_u(k - lag))

        x = np.asarray(feats, dtype=float).reshape(1, -1)
        Yhat[t] = model.predict(x)[0]

    return Yhat


def fit_and_score_extratrees_narx(
    Utr: np.ndarray,
    Ytr: np.ndarray,
    d: int,
    na: int,
    nb: int,
    base_model,
    train_ratio: float = 0.8,
):
    """
    Train on the first chunk and validate by recursive rollout on the second chunk.
    """
    U_train, Y_train, U_val, Y_val = temporal_split(Utr, Ytr, train_ratio=train_ratio)

    X_train, y_train = build_narx_supervised(U_train, Y_train, d=d, na=na, nb=nb)

    model = MultiOutputRegressor(clone(base_model))
    model.fit(X_train, y_train)

    Y_val_hat = rollout_predict(model, U_val, d=d, na=na, nb=nb)

    mse = mean_squared_error(Y_val, Y_val_hat)
    rmse = np.sqrt(mse)
    score_nmse = nmse(Y_val, Y_val_hat)

    return model, rmse, score_nmse, Y_val_hat, Y_val


def evaluate_extratrees_narx_long_term(
    Utr: np.ndarray,
    Ytr: np.ndarray,
    Uts1: np.ndarray = None,
    Uts2: np.ndarray = None,
    train_ratio: float = 0.8,
    grid_d=(0, 1),
    grid_na=(0, 1),
    grid_nb=(0, 1),
    n_estimators: int = 400,
    max_depth=None,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    save_submission: bool = False,
    submission_path: str = "submission_extratrees_narx.npz",
):
    """
    Extra Trees NARX with recursive long-term prediction.

    Workflow
    --------
    1. Temporal split on training data
    2. Grid search over (d, na, nb) using validation NMSE
    3. Refit best model on full training set
    4. Optionally generate submission predictions for Uts1 / Uts2
    5. Plot validation predictions and error

    Returns
    -------
    results_df : pd.DataFrame
    final_model : fitted model on full training set
    best_config : dict
    Y_val_hat : ndarray
    Y_val : ndarray
    Yhat1 : ndarray or None
    Yhat2 : ndarray or None
    """
    base_model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )

    search_results = []

    for d, na, nb in product(grid_d, grid_na, grid_nb):
        try:
            _, rmse_val, nmse_val, _, _ = fit_and_score_extratrees_narx(
                Utr=Utr,
                Ytr=Ytr,
                d=d,
                na=na,
                nb=nb,
                base_model=base_model,
                train_ratio=train_ratio,
            )

            search_results.append(
                {
                    "Model": "ExtraTrees_NARX_LongTerm",
                    "d": d,
                    "na": na,
                    "nb": nb,
                    "RMSE_val": rmse_val,
                    "NMSE_val": nmse_val,
                }
            )

            print(
                f"d={d}, na={na}, nb={nb} -> RMSE={rmse_val:.6f}, NMSE={nmse_val:.6f}"
            )

        except Exception as e:
            print(f"Failed for d={d}, na={na}, nb={nb}: {e}")

    results_df = (
        pd.DataFrame(search_results).sort_values("NMSE_val").reset_index(drop=True)
    )

    if results_df.empty:
        raise ValueError("No valid configuration was found during grid search.")

    best_row = results_df.iloc[0]
    best_d = int(best_row["d"])
    best_na = int(best_row["na"])
    best_nb = int(best_row["nb"])

    print("\n=== Extra Trees NARX (Long-Term) ===")
    print(f"Best config -> d={best_d}, na={best_na}, nb={best_nb}")
    print(f"Validation RMSE: {best_row['RMSE_val']:.4f}")
    print(f"Validation NMSE: {best_row['NMSE_val']:.4f}")

    # Validation predictions with best config
    _, _, _, Y_val_hat, Y_val = fit_and_score_extratrees_narx(
        Utr=Utr,
        Ytr=Ytr,
        d=best_d,
        na=best_na,
        nb=best_nb,
        base_model=base_model,
        train_ratio=train_ratio,
    )

    # Retrain on full training data
    X_full, y_full = build_narx_supervised(Utr, Ytr, d=best_d, na=best_na, nb=best_nb)
    final_model = MultiOutputRegressor(clone(base_model))
    final_model.fit(X_full, y_full)

    # Optional test predictions
    Yhat1 = None
    Yhat2 = None

    if Uts1 is not None:
        Yhat1 = rollout_predict(final_model, Uts1, d=best_d, na=best_na, nb=best_nb)

    if Uts2 is not None:
        Yhat2 = rollout_predict(final_model, Uts2, d=best_d, na=best_na, nb=best_nb)

    if save_submission and (Yhat1 is not None) and (Yhat2 is not None):
        np.savez(submission_path, Yhat1=Yhat1, Yhat2=Yhat2)
        print(f"Saved {submission_path}")

    # --- Plots: validation y1 ---
    plt.figure(figsize=(10, 3))
    plt.plot(Y_val[:, 0], label="y1 true")
    plt.plot(Y_val_hat[:, 0], label="y1 pred")
    plt.title("ExtraTrees NARX Long-Term - Validation y1")
    plt.legend()
    plt.grid()

    # --- Plots: validation y2 ---
    plt.figure(figsize=(10, 3))
    plt.plot(Y_val[:, 1], label="y2 true")
    plt.plot(Y_val_hat[:, 1], label="y2 pred")
    plt.title("ExtraTrees NARX Long-Term - Validation y2")
    plt.legend()
    plt.grid()

    # --- Zoom on first 100 steps ---
    zoom_n = min(100, len(Y_val))
    plt.figure(figsize=(10, 3))
    plt.plot(Y_val[:zoom_n, 0], label="y1 true")
    plt.plot(Y_val_hat[:zoom_n, 0], label="y1 pred")
    plt.title(f"ExtraTrees NARX Long-Term - Validation y1 (first {zoom_n} steps)")
    plt.legend()
    plt.grid()

    # --- Prediction error over time ---
    error = np.linalg.norm(Y_val - Y_val_hat, axis=1)
    plt.figure(figsize=(10, 3))
    plt.plot(error)
    plt.title("Prediction Error Over Time")
    plt.grid()

    plt.show()

    best_config = {
        "d": best_d,
        "na": best_na,
        "nb": best_nb,
    }

    return results_df, final_model, best_config, Y_val_hat, Y_val, Yhat1, Yhat2


def plot_residual_acf_pacf(y_true, y_pred, lags=20):
    """
    Plot ACF and PACF of residuals for both outputs.
    """
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


# =========================
# Example usage
# =========================

path = "../../data/raw/StudentdataNARX.npz"
data = np.load(path)

Utr = data["Utr"]
Ytr = data["Ytr"]
Uts1 = data["Uts1"]
Uts2 = data["Uts2"]

print("Utr shape:", Utr.shape)
print("Ytr shape:", Ytr.shape)
print("Uts1 shape:", Uts1.shape)
print("Uts2 shape:", Uts2.shape)

corr = np.corrcoef(Ytr.T)
print("Correlation between y1 and y2:\n", corr)

# Optional quick inspection plots
plt.figure(figsize=(10, 3))
plt.plot(Utr[:, 0], label="u1")
plt.plot(Utr[:, 1], label="u2")
plt.title("Training Inputs (Utr)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(Ytr[:, 0], label="y1")
plt.plot(Ytr[:, 1], label="y2")
plt.title("Training Outputs (Ytr)")
plt.legend()
plt.grid()
plt.show()

results, model, best_config, y_pred, y_test, Yhat1, Yhat2 = (
    evaluate_extratrees_narx_long_term(
        Utr=Utr,
        Ytr=Ytr,
        Uts1=Uts1,
        Uts2=Uts2,
        train_ratio=0.8,
        grid_d=(0, 1),
        grid_na=(0, 1),
        grid_nb=(0, 1),
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        save_submission=True,
        submission_path="submission_extratrees_narx.npz",
    )
)

print("\nTop configs:")
print(results.head(10))

plot_residual_acf_pacf(y_test, y_pred, lags=40)
