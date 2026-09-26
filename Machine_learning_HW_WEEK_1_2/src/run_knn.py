"""Part 1.3 -- Train, evaluate, and compare KNN.

Sweeps k in {1,3,5,7,9,11,15,21}, evaluates all four metrics on the test set,
reports the best k by F1, and confirms predictions/metrics match sklearn.

Run from the project root:
    python src/run_knn.py
"""

import sys
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe on all machines
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as skm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from knn import KNN
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

# Sweep k
k_values = [1, 3, 5, 7, 9, 11, 15, 21]

results = []  # list of dicts, one per k

print(f"{'k':>4} | {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} | sklearn match")
print("-" * 60)

for k in k_values:
    # your implementation
    my_knn = KNN(k=k).fit(X_train, y_train)
    my_preds = my_knn.predict(X_test)

    my_acc = accuracy(y_test, my_preds)
    my_prec = precision(y_test, my_preds)
    my_rec = recall(y_test, my_preds)
    my_f1 = f1_score(y_test, my_preds)

    # sklearn reference
    sk_knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    sk_preds = sk_knn.predict(X_test)

    sk_acc = skm.accuracy_score(y_test, sk_preds)
    sk_prec = skm.precision_score(y_test, sk_preds, zero_division=0)
    sk_rec = skm.recall_score(y_test, sk_preds, zero_division=0)
    sk_f1 = skm.f1_score(y_test, sk_preds, zero_division=0)

    # parity checks
    assert np.array_equal(
        my_preds, sk_preds
    ), f"k={k}: predictions differ from sklearn!"
    assert abs(my_acc - sk_acc) < 1e-9, f"k={k}: accuracy mismatch"
    assert abs(my_prec - sk_prec) < 1e-9, f"k={k}: precision mismatch"
    assert abs(my_rec - sk_rec) < 1e-9, f"k={k}: recall mismatch"
    assert abs(my_f1 - sk_f1) < 1e-9, f"k={k}: F1 mismatch"

    results.append({"k": k, "acc": my_acc, "prec": my_prec, "rec": my_rec, "f1": my_f1})
    print(f"{k:>4} | {my_acc:>7.4f} {my_prec:>7.4f} {my_rec:>7.4f} {my_f1:>7.4f} | OK")

# Best k by F1
best = max(results, key=lambda r: r["f1"])
print(
    f"\nBest k by F1: k={best['k']}  "
    f"(acc={best['acc']:.4f}, prec={best['prec']:.4f}, "
    f"rec={best['rec']:.4f}, F1={best['f1']:.4f})"
)

# Plot: metrics vs k
ks = [r["k"] for r in results]
accs = [r["acc"] for r in results]
precs = [r["prec"] for r in results]
recs = [r["rec"] for r in results]
f1s = [r["f1"] for r in results]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(ks, accs, marker="o", label="Accuracy", linewidth=2)
ax.plot(ks, precs, marker="s", label="Precision", linewidth=2)
ax.plot(ks, recs, marker="^", label="Recall", linewidth=2)
ax.plot(ks, f1s, marker="D", label="F1", linewidth=2, color="crimson")

# mark the best k
ax.axvline(
    x=best["k"],
    color="gray",
    linestyle="--",
    alpha=0.6,
    label=f"Best k={best['k']} (F1={best['f1']:.4f})",
)

ax.set_xlabel("k (number of neighbors)", fontsize=12)
ax.set_ylabel("Metric value", fontsize=12)
ax.set_title(
    "KNN: Classification Metrics vs k\n(Breast Cancer dataset, test set)", fontsize=13
)
ax.set_xticks(ks)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.85, 1.01)

plt.tight_layout()

out_path = os.path.join(
    os.path.dirname(__file__), "..", "figures", "knn_metrics_vs_k.png"
)
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved -> {os.path.abspath(out_path)}")
