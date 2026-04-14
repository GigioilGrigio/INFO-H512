import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_pickle("../data/interim/training_df.pkl")


def plot_inputs_outputs(df):
    """
    Plots input signals (u1, u2) and output signals (y1, y2)
    in two separate figures.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['u1', 'u2', 'y1', 'y2']
                           and a timestamp index or column.
    """

    # --- Plot inputs ---
    plt.figure()
    plt.plot(df.index, df["u1"], label="u1")
    plt.plot(df.index, df["u2"], label="u2")
    plt.title("Input Signals")
    plt.xlabel("Time")
    plt.ylabel("Inputs")
    plt.legend()
    plt.grid()

    # --- Plot outputs ---
    plt.figure()
    plt.plot(df.index, df["y1"], label="y1")
    plt.plot(df.index, df["y2"], label="y2")
    plt.title("Output Signals")
    plt.xlabel("Time")
    plt.ylabel("Outputs")
    plt.legend()
    plt.grid()

    plt.show()


plot_inputs_outputs(df)


def visualize_time_series_correlations(
    df, variables=["u1", "u2", "y1", "y2"], max_lag=30, integrate=False
):
    """
    Plot ACF, PACF, and CCF for time series variables.

    Parameters:
        df (pd.DataFrame): must contain variables
        variables (list): variables to analyze
        max_lag (int): max lag
        integrate (bool): if True, use cumulative sum of signals
    """

    df_proc = df.copy()

    # --- Optional integration ---
    if integrate:
        df_proc[variables] = df_proc[variables].cumsum()
        print("Using integrated (cumulative sum) series")

    # --- ACF & PACF ---
    for var in variables:
        series = df_proc[var].values

        fig, axes = plt.subplots(1, 2, figsize=(10, 3))

        plot_acf(series, lags=max_lag, ax=axes[0])
        axes[0].set_title(f"ACF - {var}")

        plot_pacf(series, lags=max_lag, ax=axes[1])
        axes[1].set_title(f"PACF - {var}")

        plt.tight_layout()

    # --- Cross-correlation function ---
    def cross_corr(x, y, max_lag):
        lags = range(0, max_lag)
        corr = [np.corrcoef(x[lag:], y[:-lag] if lag != 0 else y)[0, 1] for lag in lags]
        return lags, corr

    # --- CCF: inputs vs outputs ---
    inputs = ["u1", "u2"]
    outputs = ["y1", "y2"]

    for u in inputs:
        for y in outputs:
            x = df_proc[y].values
            u_series = df_proc[u].values

            lags, corr = cross_corr(x, u_series, max_lag)

            plt.figure(figsize=(8, 3))
            plt.stem(lags, corr)
            plt.title(f"CCF: {y} vs {u}")
            plt.xlabel("Lag")
            plt.ylabel("Correlation")
            plt.grid()

    plt.show()


visualize_time_series_correlations(df, max_lag=100)
