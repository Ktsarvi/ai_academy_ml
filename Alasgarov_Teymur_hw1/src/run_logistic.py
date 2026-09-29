"""Part 2.2 -- Train, evaluate, and compare Logistic Regression.

Trains logistic regression from scratch using batch gradient descent on
binary cross-entropy loss with the breast cancer dataset, computes classification
metrics (accuracy, precision, recall, F1) using metrics from scratch, compares against
sklearn's LogisticRegression, and plots the training loss curve.

Run from the project root:
    python src/run_logistic.py
"""

import sys
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend -- safe on all machines
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import sklearn.metrics as skm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from logistic_regression import LogisticRegression, sigmoid
from metrics import accuracy, precision, recall, f1_score

np.random.seed(42)

# Load & prepare data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize - fit scaler on TRAIN only
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train from-scratch Logistic Regression via Gradient Descent
lr = 0.1
epochs = 1000
my_model = LogisticRegression(lr=lr, epochs=epochs).fit(X_train, y_train)
my_preds = my_model.predict(X_test)

# Compute classification metrics using our scratch metrics.py
my_acc = accuracy(y_test, my_preds)
my_prec = precision(y_test, my_preds)
my_rec = recall(y_test, my_preds)
my_f1 = f1_score(y_test, my_preds)

# Sklearn reference (unregularized, C=np.inf disables the L2 penalty
# without the deprecation warning that penalty=None now raises)
sk_model = SklearnLogisticRegression(C=np.inf, random_state=42, max_iter=5000).fit(
    X_train, y_train
)
sk_preds = sk_model.predict(X_test)

sk_acc = skm.accuracy_score(y_test, sk_preds)
sk_prec = skm.precision_score(y_test, sk_preds, zero_division=0)
sk_rec = skm.recall_score(y_test, sk_preds, zero_division=0)
sk_f1 = skm.f1_score(y_test, sk_preds, zero_division=0)

n_diff = int(np.sum(my_preds != sk_preds))


# Helper to compute BCE loss for arbitrary (w, b), used to compare the two
# solutions on equal footing (train-set loss, not just test accuracy).
# Reuses the same sigmoid the model itself trains with
# rather than redefining it, so the two never drift out of sync.
def bce(w, b, X, y):
    p = sigmoid(X @ w + b)
    return -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))


my_train_loss = bce(my_model.w, my_model.b, X_train, y_train)
sk_train_loss = bce(sk_model.coef_[0], sk_model.intercept_[0], X_train, y_train)
max_w_diff = float(np.max(np.abs(my_model.w - sk_model.coef_[0])))

# Display Comparison Table
print("=" * 68)
print(f"{'Model':<30} | {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("-" * 68)
print(
    f"{'From-scratch GD (1000 epochs)':<30} | "
    f"{my_acc:>7.4f} {my_prec:>7.4f} {my_rec:>7.4f} {my_f1:>7.4f}"
)
print(
    f"{'Sklearn (C=inf, L-BFGS)':<30} | "
    f"{sk_acc:>7.4f} {sk_prec:>7.4f} {sk_rec:>7.4f} {sk_f1:>7.4f}"
)
print("=" * 68)
print(f"Test predictions differing from sklearn: {n_diff} / {len(y_test)}")
print(f"Train-set BCE loss  -- mine: {my_train_loss:.6f}  sklearn: {sk_train_loss:.6f}")
print(f"Max |weight| difference: {max_w_diff:.2f}")

print("\n--- Discrepancy & Gap Explanation ---")
print(
    "1. After standardization the breast cancer classes are nearly linearly"
    "\n   separable. For a (near-)separable dataset, the unregularized cross-entropy"
    "\n   objective has no finite minimizer: pushing ||w|| -> infinity keeps"
    "\n   sharpening the sigmoid and keeps lowering the loss."
    "\n2. Sklearn's L-BFGS runs to a strict convergence tolerance and travels much"
    "\n   further along this diverging path, reaching a much lower train-set loss"
    "\n   and much larger weight magnitudes (see numbers above)."
    "\n3. Our fixed-epoch-budget GD stops earlier along the same trajectory, so the"
    "\n   two solutions disagree on a handful of test predictions -- not because GD"
    "\n   failed to converge to 'the' optimum, but because there isn't a finite one"
    "\n   for this (near-)separable problem. Both solutions still score highly on"
    "\n   the test metrics; which one is higher can vary by dataset split."
)

# Plot: Training Loss Curve
figures_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(figures_dir, exist_ok=True)
out_path = os.path.join(figures_dir, "logistic_loss.png")

fig, ax = plt.subplots(figsize=(8, 5))
epochs_range = np.arange(1, len(my_model.loss_history) + 1)
ax.plot(
    epochs_range,
    my_model.loss_history,
    color="#0B3D91",
    linewidth=2,
    label="Binary Cross-Entropy Loss",
)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss (Binary Cross-Entropy)", fontsize=12)
ax.set_title(
    "Logistic Regression: Training Loss vs. Epochs\n(Breast Cancer dataset, Batch GD)",
    fontsize=13,
)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=11)

# Annotate final loss
final_loss = my_model.loss_history[-1]
ax.annotate(
    f"Final Loss: {final_loss:.4f}",
    xy=(len(my_model.loss_history), final_loss),
    xytext=(
        len(my_model.loss_history) * 0.65,
        final_loss + (my_model.loss_history[0] - final_loss) * 0.15,
    ),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
    fontsize=10,
    fontweight="bold",
)

plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close(fig)

print(f"\nPlot saved -> {os.path.abspath(out_path)}")
