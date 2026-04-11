import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
