# HW1 (Weeks 1–2) — Complete Roadmap

**MLE-AI-201 · Machine Learning · AI Academy, National AI Center · Cohort II · Fall 2026**
**Student: Alasgarov Teymur**
**Due: October 4, 2026 at 23:59 (UTC+4) · 100 pts + 10 bonus**

---

## ⚠️ Penalty-Avoidance Rules (read first)

| Rule                                         | What to do                                                                 |
| -------------------------------------------- | -------------------------------------------------------------------------- |
| No `for` loop over training points in KNN    | Use vectorized NumPy (`@`, broadcasting) — **−4 pts** if violated          |
| Metrics must match sklearn within 1e-9       | Test with `assert abs(yours - sklearn) < 1e-9` — **−4 pts** if violated    |
| No normal equation or sklearn fit in Part 2  | Gradient descent only — **−10 pts** if violated                            |
| No sklearn in `src/` except for data loading | Keep sklearn only in run/eval scripts                                      |
| Seed everything                              | `np.random.seed(42)` at the top of every script — **up to −5 pts**         |
| Scaler fit on train split only               | `scaler.fit(X_train)`, then `.transform(X_train)` and `.transform(X_test)` |
| Loss curves must be monotonically decreasing | Tune `lr` if diverging (try `lr=0.01` or scale features first)             |
| Report ≤ 6 pages                             | Be concise; 1 figure per task minimum                                      |
| Declare AI use in `pledge.txt`               | Name the tool, what for, which parts, how you checked                      |

---

## 🗺️ Step-by-Step Roadmap

### ✅ PHASE 0 — Environment

- [x] `pip install -r requirements.txt`
- [x] Verify: `python -c "import numpy, matplotlib, sklearn, pytest; print('OK')"`

---

### PHASE 1 — `src/knn.py` · Part 1.1 · 20 pts

**Task: implement the `KNN` class with fully vectorized NumPy (no Python loops over training points).**

- [x] **`fit(X, y)`** — store `X_train = X.astype(float)`, `y_train = y.astype(int)`, return `self`
- [x] **`_distances(X)`** — vectorized pairwise Euclidean distance matrix, shape `(n_queries, n_train)`:
  ```python
  # Use: ||x - z||² = ||x||² - 2·X·Xₜᵀ + ||Xₜ||²
  sq_X  = np.sum(X**2, axis=1, keepdims=True)             # (n_q, 1)
  sq_Xt = np.sum(self.X_train**2, axis=1, keepdims=True)  # (n_t, 1)
  cross = X @ self.X_train.T                               # (n_q, n_t)
  dist2 = sq_X - 2*cross + sq_Xt.T                        # (n_q, n_t)
  return np.sqrt(np.maximum(dist2, 0))                     # clip numerical negatives
  ```
- [x] **`predict(X)`** — majority-vote classification:
  ```python
  D = self._distances(X)                                      # (n_q, n_t)
  k_idx   = np.argpartition(D, self.k, axis=1)[:, :self.k]   # k nearest indices
  k_labels = self.y_train[k_idx]                              # (n_q, k)
  return np.array([np.bincount(row).argmax() for row in k_labels])
  ```

> **Self-check:** `grep "for " src/knn.py` must not show a loop over training points.

---

### PHASE 2 — `src/metrics.py` · Part 1.2 · 20 pts

**Task: implement the 4 metrics purely from confusion-matrix counts.**

```python
# Pattern for each function:
tp = np.sum((y_pred == pos) & (y_true == pos))
fp = np.sum((y_pred == pos) & (y_true != pos))
fn = np.sum((y_pred != pos) & (y_true == pos))
```

- [x] **`accuracy`**: `float(np.mean(y_true == y_pred))` — no edge case needed
- [x] **`precision`**: `tp / (tp + fp)` → `0.0` if denominator is 0
- [x] **`recall`**: `tp / (tp + fn)` → `0.0` if denominator is 0
- [x] **`f1_score`**: `2*p*r / (p+r)` → `0.0` if `p + r == 0`

---

### PHASE 3 — `src/run_knn.py` · Part 1.3 · 10 pts

**Task: sweep k, produce plot, confirm parity with sklearn.**

- [x] Load breast cancer, `train_test_split(random_state=42)`, standardize (fit on train)
- [x] Loop `k ∈ {1, 3, 5, 7, 9, 11, 15, 21}`:
  - Fit your `KNN(k)` → predict → compute your 4 metrics
  - Fit `sklearn KNeighborsClassifier(k)` → `assert np.array_equal(yours, sklearn_preds)`
  - `assert abs(your_metric - sklearn_metric) < 1e-9` for each metric
- [x] Print best k by F₁
- [x] **Plot** (required by rubric):
  - x-axis: k values, y-axis: metric value
  - 4 lines: accuracy, precision, recall, F₁
  - Save → `figures/knn_metrics_vs_k.png`

---

### PHASE 4 — `src/linear_regression.py` · Part 2.1 · 25 pts

**Task: batch gradient descent on MSE, diabetes dataset.**

- [x] **`fit(X, y)`**:
  ```python
  np.random.seed(42)
  N, p = X.shape
  self.w = np.zeros(p);  self.b = 0.0
  for _ in range(self.epochs):
      y_hat  = X @ self.w + self.b
      loss   = np.mean((y_hat - y)**2)
      self.loss_history.append(loss)
      grad_w = (2/N) * X.T @ (y_hat - y)
      grad_b = (2/N) * np.sum(y_hat - y)
      # BONUS ---
      if self.penalty == "l1":  grad_w += self.lam * np.sign(self.w)
      elif self.penalty == "l2": grad_w += 2 * self.lam * self.w
      self.w -= self.lr * grad_w;  self.b -= self.lr * grad_b
  return self
  ```
- [x] **`predict(X)`**: `return X @ self.w + self.b`

**Create `src/run_linear.py`:**

- [x] Load diabetes, standardize, split (`random_state=42`)
- [x] Fit `LinearRegression(lr=0.1, epochs=1000)`, plot loss → `figures/linear_loss.png`
- [x] Compute test MSE and R² (`1 - SS_res/SS_tot`)
- [x] Compare vs `sklearn.linear_model.LinearRegression`, explain gap (GD vs closed-form)

---

### PHASE 5 — `src/logistic_regression.py` · Part 2.2 · 25 pts

**Task: gradient descent on binary cross-entropy, breast cancer dataset.**

- [x] **`sigmoid(z)`** — numerically stable:
  ```python
  return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
  ```
- [x] **`fit(X, y)`**:
  ```python
  np.random.seed(42)
  N, p = X.shape
  self.w = np.zeros(p);  self.b = 0.0
  for _ in range(self.epochs):
      p_hat  = sigmoid(X @ self.w + self.b)
      loss   = -np.mean(y*np.log(p_hat+1e-15) + (1-y)*np.log(1-p_hat+1e-15))
      self.loss_history.append(loss)
      grad_w = (1/N) * X.T @ (p_hat - y)
      grad_b = (1/N) * np.sum(p_hat - y)
      # BONUS ---
      if self.penalty == "l1":  grad_w += self.lam * np.sign(self.w)
      elif self.penalty == "l2": grad_w += 2 * self.lam * self.w
      self.w -= self.lr * grad_w;  self.b -= self.lr * grad_b
  return self
  ```
- [x] **`predict_proba(X)`**: `return sigmoid(X @ self.w + self.b)`
- [x] **`predict(X, threshold=0.5)`**: `return (self.predict_proba(X) >= threshold).astype(int)`

**Create `src/run_logistic.py`:**

- [x] Load breast cancer, standardize, split (`random_state=42`)
- [x] Fit `LogisticRegression(lr=0.1, epochs=1000)`, plot loss → `figures/logistic_loss.png`
- [x] Compute accuracy, precision, recall, F₁ (reuse your `metrics.py`)
- [x] Compare vs `sklearn.linear_model.LogisticRegression`, explain gap

---

### PHASE 6 — Bonus: Regularization · +10 pts

Gradient terms already embedded in Phases 4 & 5 above.

**Create `src/run_regularization.py`:**

- [x] For **L1** and **L2** separately, loop `λ ∈ {0, 0.001, 0.01, 0.1, 1.0, 10.0}`:
  - Fit model with `penalty="l1"/"l2"`, `lam=λ`
  - Record: test metric, `np.linalg.norm(w, 1)`, `np.linalg.norm(w, 2)`
- [x] **Plot L1** (`figures/reg_l1.png`): λ on x-axis (log scale), dual y-axis: ‖w‖₁ + test metric
- [x] **Plot L2** (`figures/reg_l2.png`): same layout with ‖w‖₂
- [x] In report: L1 → some weights → ~0 (sparse); L2 → all shrink smoothly

---

### PHASE 7 — pytest

```bash
cd d:\.vscode\ai_academy_ml\Machine_learning_HW_WEEK_1_2
pytest -v
```

- [x] `tests/test_knn.py` — all green
- [x] `tests/test_metrics.py` — all green
- [x] Fix all failures before writing the report

---

### PHASE 8 — Report (`report_template.tex` → `report.pdf`) · max 6 pages

| Section               | Required content                                                                   |
| --------------------- | ---------------------------------------------------------------------------------- |
| Header                | Name: Alasgarov Teymur, Student ID, Date                                           |
| **1.1 KNN**           | Describe vectorized distance trick                                                 |
| **1.2 Metrics**       | Formulas used, zero-denominator handling                                           |
| **1.3 Evaluation**    | Table: k vs acc/prec/rec/F₁ · Figure: `knn_metrics_vs_k.png` · Best k by F₁        |
| **2.1 Linear Reg.**   | Figure: `linear_loss.png` · Test MSE + R² · sklearn comparison + gap explanation   |
| **2.2 Logistic Reg.** | Figure: `logistic_loss.png` · All 4 metrics · sklearn comparison + gap explanation |
| **Bonus**             | Figures: `reg_l1.png`, `reg_l2.png` · Explain sparsity (L1) vs shrinkage (L2)      |
| AI Use                | Tool name, tasks it assisted, how you verified                                     |

```bash
pdflatex report_template.tex && pdflatex report_template.tex
```

- [ ] Verify PDF ≤ 6 pages

---

### PHASE 9 — `pledge.txt`

- [x] Fill in your name: **Alasgarov Teymur** and today's date
- [x] Declare AI use: tool name, which parts it helped with, how you verified

---

### PHASE 10 — Submission

Final zip contents:

```
Alasgarov_Teymur_hw1.zip
├── src/
│   ├── knn.py
│   ├── metrics.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── run_knn.py
│   ├── run_linear.py
│   ├── run_logistic.py
│   └── run_regularization.py   ← bonus
├── figures/
│   ├── knn_metrics_vs_k.png
│   ├── linear_loss.png
│   ├── logistic_loss.png
│   ├── reg_l1.png               ← bonus
│   └── reg_l2.png               ← bonus
├── report.pdf
├── pledge.txt
└── requirements.txt
```

```powershell
Compress-Archive -Path .\* -DestinationPath ..\Alasgarov_Teymur_hw1.zip
```

---

## ⏱️ Time Estimate

| Phase     | Task                             | Est. Time  |
| --------- | -------------------------------- | ---------- |
| 1         | KNN (vectorized)                 | 40 min     |
| 2         | Metrics                          | 20 min     |
| 3         | KNN eval + plot                  | 30 min     |
| 4         | Linear regression + run script   | 45 min     |
| 5         | Logistic regression + run script | 45 min     |
| 6         | Bonus: regularization            | 60 min     |
| 7         | pytest debugging                 | 15 min     |
| 8         | Report writing                   | 90 min     |
| 9–10      | Pledge + zip + upload            | 10 min     |
| **Total** |                                  | **~6 hrs** |
