from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
file_path = DATA_DIR / "StudentdataNARX.npz"

data = np.load(file_path)

Utr = data["Utr"]
Ytr = data["Ytr"]
Uts1 = data["Uts1"]
Uts2 = data["Uts2"]

plt.figure()

plt.plot(Ytr[:, 0], label="y1")
plt.plot(Ytr[:, 1], label="y2")

plt.xlabel("Time step")
plt.ylabel("Output")
plt.title("Training Output (Ytr)")
plt.legend()

plt.show()


t = np.arange(Ytr.shape[0])  # time axis

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

# --- Outputs ---
axs[0].plot(t, Ytr[:, 0], label="y1")
axs[0].plot(t, Ytr[:, 1], label="y2")
axs[0].set_title("Outputs (Ytr)")
axs[0].legend()

# --- Input u1 ---
axs[1].plot(t, Utr[:, 0], label="u1")
axs[1].set_title("Input u1")

# --- Input u2 ---
axs[2].plot(t, Utr[:, 1], label="u2")
axs[2].set_title("Input u2")

plt.xlabel("Time step")
plt.tight_layout()
plt.show()
