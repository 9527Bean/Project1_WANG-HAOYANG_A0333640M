
# Project1 Task1-3

---



## 1) Environment

- Python 3.9+  
- Packages: `numpy`, `pandas`, `matplotlib`

Install (Windows PowerShell / CMD):

```bash
pip install numpy pandas matplotlib

Task1 — Ridge Regression (fine grid, beautified plots)

This task reproduces the paper's "virtue of complexity" results under partial observability using **Ridge** (including ridgeless `z=0`), on the same DGP as the paper. We sweep a **fine grid** of shrinkage `z` and observed complexity `c_q`.

## 2) Task1 Output：
T1_fig1_R2_vs_cq_fine_v2.png
T1_fig2_E_Rpi_vs_cq_fine_v2.png
T1_fig3_Sharpe_vs_cq_fine_v2.png
T1_fig4_beta_norm2_fine_v2.png
Data (CSV)：task1_metrics_finegrid.csv



---

# Task2 — LASSO (fine grid, beautified plots)

This task repeats Task1 under the **same data and split** but replaces Ridge with **LASSO** (ISTA implementation). We sweep a **log-spaced fine grid** of λ and the same `c_q` grid.

---

## 1) Environment

- Python 3.9+  
- Packages: `numpy`, `pandas`, `matplotlib`

Install:

```bash
pip install numpy pandas matplotlib


## 2) Task2 Output：

T2_fig1_R2_vs_cq_LASSO_fine_v2.png
T2_fig2_E_Rpi_vs_cq_LASSO_fine_v2.png
T2_fig3_Sharpe_vs_cq_LASSO_fine_v2.png
T2_fig4_beta_norm2_LASSO_fine_v2.png
Data (CSV)：task2_lasso_metrics_finegrid.csv



# Task 3 — Theory + On-Sample Evidence Model Selection & Prediction

This project selects **Ridge vs. Lasso** per pair (A/B/C) using:
- **Theory-guided priors** (sparsity vs. collinearity diagnostics), and
- **Train-only, time-respecting rolling CV** (Sharpe as primary, plus expected return, paper-style R², stability),
then **fits on the full training set** and outputs **test predictions**.

All outputs are written to:
`C:\Users\13970\Desktop\QF结果图` (created automatically).

---

## 1) Files & Data Schema

**Main script**
- `task3_predict_theory_evidence_plots.py` — selection + predictions + optional plots

**Input data (CSV)**
- `pairA_train.csv`, `pairA_test_features.csv`
- `pairB_train.csv`, `pairB_test_features.csv`
- `pairC_train.csv`, `pairC_test_features.csv`

**Schema**
- Train: `t, feature1..featureP, return`
- Test : `t, feature1..featureP`  
Feature *sets* must match exactly (the script checks and then reorders test columns to match train).

---

## 2) Environment Setup

> Recommended: Python 3.10 or 3.11

**Windows (PowerShell)**
```powershell
cd path\to\project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
SAVE_DIR = r"C:\Users\13970\Desktop\QF结果图"  # Output folder

PAIR_PATHS = {
    "A": {
        "train": r"./data/pairA_train.csv",
        "test":  r"./data/pairA_test_features.csv",
        "out":   os.path.join(SAVE_DIR, "A0333640M_predictions_A.csv"),
    },
    "B": {
        "train": r"./data/pairB_train.csv",
        "test":  r"./data/pairB_test_features.csv",
        "out":   os.path.join(SAVE_DIR, "A0333640M_predictions_B.csv"),
    },
    "C": {
        "train": r"./data/pairC_train.csv",
        "test":  r"./data/pairC_test_features.csv",
        "out":   os.path.join(SAVE_DIR, "A0333640M_predictions_C.csv"),
    },
}

MAKE_PLOTS = True  # set False to skip plot generation
The script uses a fixed random seed (SEED = 7) for reproducibility.
