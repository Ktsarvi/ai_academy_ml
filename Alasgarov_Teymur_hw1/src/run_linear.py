"""Part 2.1 -- Train, evaluate, and compare Linear Regression.

Trains linear regression from scratch using batch gradient descent on the
diabetes dataset, computes test MSE and R^2, compares against sklearn's
closed-form LinearRegression, and plots the training loss curve.

Run from the project root:
    python src/run_linear.py
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
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
import sklearn.metrics as skm

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

# Train from-scratch Linear Regression via Gradient Descent
lr = 0.1
epochs = 1000
my_lr = LinearRegression(lr=lr, epochs=epochs).fit(X_train, y_train)
my_preds = my_lr.predict(X_test)

# Compute metrics from scratch
ss_res = np.sum((y_test - my_preds) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
my_mse = float(np.mean((y_test - my_preds) ** 2))
my_r2 = float(1.0 - (ss_res / ss_tot))

# Sklearn reference (exact closed-form solution via OLS / SVD)
sk_lr = SklearnLinearRegression().fit(X_train, y_train)
sk_preds = sk_lr.predict(X_test)

sk_mse = float(skm.mean_squared_error(y_test, sk_preds))
sk_r2 = float(skm.r2_score(y_test, sk_preds))

# Display Comparison
print("=" * 65)
print(f"{'Model':<30} | {'Test MSE':>12} | {'Test R^2':>12}")
print("-" * 65)
print(f"{'From-scratch GD (1000 epochs)':<30} | {my_mse:>12.4f} | {my_r2:>12.4f}")
print(f"{'Sklearn (Closed-form OLS)':<30} | {sk_mse:>12.4f} | {sk_r2:>12.4f}")
print("=" * 65)

mse_diff = abs(my_mse - sk_mse)
r2_diff = abs(my_r2 - sk_r2)
print(f"\nDiscrepancy (Absolute Difference):")
print(f"  MSE Gap : {mse_diff:.6f}")
print(f"  R^2 Gap : {r2_diff:.6f}")

print("\n--- Gap Explanation ---")
print(
    "1. Sklearn solves the Ordinary Least Squares (OLS) problem using the analytical"
    "\n   closed-form solution via LAPACK/SVD (normal equation equivalent),"
    "\n   finding the exact global minimum in one step."
    "\n2. Our implementation uses Batch Gradient Descent (GD), an iterative first-order"
    "\n   optimization algorithm that asymptotically approaches the minimum."
    "\n3. With standardized features, lr=0.1, and 1000 epochs, GD converges extremely close"
    "\n   to the global optimum, leaving only a negligible numerical gap."
)

# Plot: Training Loss Curve
figures_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(figures_dir, exist_ok=True)
out_path = os.path.join(figures_dir, "linear_loss.png")

fig, ax = plt.subplots(figsize=(8, 5))
epochs_range = np.arange(1, len(my_lr.loss_history) + 1)
ax.plot(
    epochs_range,
    my_lr.loss_history,
    color="#0B3D91",
    linewidth=2,
    label="Train MSE Loss",
)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12)
ax.set_title(
    "Linear Regression: Training Loss vs. Epochs\n(Diabetes dataset, Batch GD)",
    fontsize=13,
)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=11)

# Annotate final loss
final_loss = my_lr.loss_history[-1]
ax.annotate(
    f"Final Loss: {final_loss:.2f}",
    xy=(len(my_lr.loss_history), final_loss),
    xytext=(
        len(my_lr.loss_history) * 0.7,
        final_loss + (my_lr.loss_history[0] - final_loss) * 0.1,
    ),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
    fontsize=10,
    fontweight="bold",
)

plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close(fig)

print(f"\nPlot saved -> {os.path.abspath(out_path)}")
