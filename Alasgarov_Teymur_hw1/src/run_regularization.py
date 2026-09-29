"""Part 2 (Bonus) -- Regularization (L1 Lasso and L2 Ridge).

Investigates the effect of L1 and L2 regularization on Linear Regression
using the diabetes dataset across lambda values in {0, 0.001, 0.01, 0.1, 1.0, 5.0}.
Computes weight norms (||w||_1, ||w||_2), sparsity, test MSE, and test R^2.
Generates comparison tables and dual-axis plots:
    - figures/reg_l1.png (L1 weight norm & test performance vs lambda)
    - figures/reg_l2.png (L2 weight norm & test performance vs lambda)

Note on the lambda range: the L2 weight update, isolating the
regularization term, is w <- w * (1 - 2*lr*lam). This is only stable
when |1 - 2*lr*lam| < 1, i.e. lam < 1/(2*lr). With lr=0.1 that bound is
lam < 5, so 5.0 (not 10.0) is used as the top of the sweep -- lam=10.0
with lr=0.1 makes the L2 weights diverge to overflow.

Run from the project root:
    python src/run_regularization.py
"""

import sys
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend -- safe on all machines
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from linear_regression import LinearRegression

np.random.seed(42)

# Load & prepare data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize - fit scaler on TRAIN only
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameters
lambdas = [0.0, 0.001, 0.01, 0.1, 1.0, 5.0]
lr = 0.1
epochs = 1000

# Containers for results
l1_results = []
l2_results = []


def evaluate_regularization(penalty: str, lam_values: list[float]):
    results = []
    for lam in lam_values:
        # None when lam == 0.0 to match unregularized baseline
        actual_penalty = None if lam == 0.0 else penalty
        model = LinearRegression(
            lr=lr, epochs=epochs, penalty=actual_penalty, lam=lam
        ).fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = float(np.mean((y_test - preds) ** 2))
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_res = np.sum((y_test - preds) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot))

        w_norm1 = float(np.linalg.norm(model.w, 1))
        w_norm2 = float(np.linalg.norm(model.w, 2))
        near_zero_weights = int(np.sum(np.abs(model.w) < 0.5))

        results.append(
            {
                "lambda": lam,
                "model": model,
                "weights": model.w.copy(),
                "norm1": w_norm1,
                "norm2": w_norm2,
                "mse": mse,
                "r2": r2,
                "zeros": near_zero_weights,
            }
        )
    return results


print("=" * 75)
print("Evaluating L1 (Lasso) Regularization across Lambdas")
print("=" * 75)
l1_results = evaluate_regularization("l1", lambdas)
print(
    f"{'Lambda':>8} | {'||w||_1':>10} | {'||w||_2':>10} | {'Test MSE':>10} | {'Test R^2':>10} | {'|w_i|<0.5':>10}"
)
print("-" * 75)
for r in l1_results:
    print(
        f"{r['lambda']:>8.3f} | {r['norm1']:>10.4f} | {r['norm2']:>10.4f} | "
        f"{r['mse']:>10.2f} | {r['r2']:>10.4f} | {r['zeros']:>10d}/{len(r['weights'])}"
    )

print("\n" + "=" * 75)
print("Evaluating L2 (Ridge) Regularization across Lambdas")
print("=" * 75)
l2_results = evaluate_regularization("l2", lambdas)
print(
    f"{'Lambda':>8} | {'||w||_1':>10} | {'||w||_2':>10} | {'Test MSE':>10} | {'Test R^2':>10} | {'|w_i|<0.5':>10}"
)
print("-" * 75)
for r in l2_results:
    print(
        f"{r['lambda']:>8.3f} | {r['norm1']:>10.4f} | {r['norm2']:>10.4f} | "
        f"{r['mse']:>10.2f} | {r['r2']:>10.4f} | {r['zeros']:>10d}/{len(r['weights'])}"
    )

# Sparsity / Weight comparison for report
print("\n" + "=" * 75)
print("Weight Vector Comparison: Unregularized vs L1 (lam=1.0) vs L2 (lam=1.0)")
print("=" * 75)
w_unreg = l1_results[0]["weights"]
w_l1 = next(r["weights"] for r in l1_results if r["lambda"] == 1.0)
w_l2 = next(r["weights"] for r in l2_results if r["lambda"] == 1.0)

print(
    f"{'Feature':<10} | {'Unreg w':>12} | {'L1 (lam=1.0)':>14} | {'L2 (lam=1.0)':>14}"
)
print("-" * 60)
for idx in range(len(w_unreg)):
    print(
        f"Feature {idx:<2} | {w_unreg[idx]:>12.4f} | {w_l1[idx]:>14.4f} | {w_l2[idx]:>14.4f}"
    )

# Ensure figures folder exists
figures_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(figures_dir, exist_ok=True)

# -------------------------------------------------------------
# Plot 1: L1 Regularization (Lasso)
# -------------------------------------------------------------
lam_labels = [str(lam) for lam in lambdas]
x_indices = np.arange(len(lambdas))

fig, ax1 = plt.subplots(figsize=(8, 5))
color_norm = "#8B0000"  # Crimson / Dark Red
color_metric = "#0B3D91"  # Navy Blue

ax1.set_xlabel(r"Regularization Strength $\lambda$", fontsize=12)
ax1.set_ylabel(r"$L_1$ Weight Norm $\|\mathbf{w}\|_1$", color=color_norm, fontsize=12)
line1 = ax1.plot(
    x_indices,
    [r["norm1"] for r in l1_results],
    color=color_norm,
    marker="o",
    linewidth=2,
    label=r"$\|\mathbf{w}\|_1$ Norm",
)
ax1.tick_params(axis="y", labelcolor=color_norm)
ax1.set_xticks(x_indices)
ax1.set_xticklabels(lam_labels)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
ax2.set_ylabel(r"Test $R^2$ Score", color=color_metric, fontsize=12)
line2 = ax2.plot(
    x_indices,
    [r["r2"] for r in l1_results],
    color=color_metric,
    marker="s",
    linewidth=2,
    linestyle="--",
    label=r"Test $R^2$",
)
ax2.tick_params(axis="y", labelcolor=color_metric)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center right", fontsize=10)
plt.title(
    r"L1 Regularization (Lasso): $\|\mathbf{w}\|_1$ and Test $R^2$ vs $\lambda$"
    "\n(Diabetes dataset, Batch GD)",
    fontsize=13,
)
plt.tight_layout()
l1_plot_path = os.path.join(figures_dir, "reg_l1.png")
plt.savefig(l1_plot_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved -> {os.path.abspath(l1_plot_path)}")

# -------------------------------------------------------------
# Plot 2: L2 Regularization (Ridge)
# -------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(8, 5))
color_norm = "#2E7D32"  # Forest Green
color_metric = "#0B3D91"  # Navy Blue

ax1.set_xlabel(r"Regularization Strength $\lambda$", fontsize=12)
ax1.set_ylabel(r"$L_2$ Weight Norm $\|\mathbf{w}\|_2$", color=color_norm, fontsize=12)
line1 = ax1.plot(
    x_indices,
    [r["norm2"] for r in l2_results],
    color=color_norm,
    marker="o",
    linewidth=2,
    label=r"$\|\mathbf{w}\|_2$ Norm",
)
ax1.tick_params(axis="y", labelcolor=color_norm)
ax1.set_xticks(x_indices)
ax1.set_xticklabels(lam_labels)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
ax2.set_ylabel(r"Test $R^2$ Score", color=color_metric, fontsize=12)
line2 = ax2.plot(
    x_indices,
    [r["r2"] for r in l2_results],
    color=color_metric,
    marker="^",
    linewidth=2,
    linestyle="--",
    label=r"Test $R^2$",
)
ax2.tick_params(axis="y", labelcolor=color_metric)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center right", fontsize=10)
plt.title(
    r"L2 Regularization (Ridge): $\|\mathbf{w}\|_2$ and Test $R^2$ vs $\lambda$"
    "\n(Diabetes dataset, Batch GD)",
    fontsize=13,
)
plt.tight_layout()
l2_plot_path = os.path.join(figures_dir, "reg_l2.png")
plt.savefig(l2_plot_path, dpi=150)
plt.close(fig)
print(f"Plot saved -> {os.path.abspath(l2_plot_path)}")
