# MLE-AI-201 Machine Learning — Homework 1 (Weeks 1–2)

**KNN & Metrics, Linear & Logistic Regression from scratch.**

Read `statement.pdf` for the full assignment. This repo is the starter kit.

## What you do
1. **Part 1 — KNN & Metrics** (50 pts): complete `src/knn.py` and
   `src/metrics.py`; train and evaluate on the breast cancer dataset.
2. **Part 2 — Regression** (50 pts): complete `src/linear_regression.py`
   (diabetes) and `src/logistic_regression.py` (breast cancer), both by
   gradient descent.
3. **Bonus** (+10): add L1 (+5) and/or L2 (+5) regularization.

All datasets are built into scikit-learn — nothing to download.

## Layout
```
statement.pdf / statement.tex   assignment
report_template.tex             fill in, compile to report.pdf
src/                            your implementations (stubs provided)
tests/                          pytest sanity checks vs scikit-learn
figures/                        put your plots here
pledge.txt                      sign it; declare AI use
```

## Run
```bash
pip install -r requirements.txt
pytest                 # sanity-check your KNN and metrics vs sklearn
tectonic report_template.tex   # or: pdflatex report_template.tex (twice)
```

## Submit
A single `.zip` named `lastname_firstname_hw1.zip` on Moodle, containing
your code, `report.pdf`, and a signed `pledge.txt`.
