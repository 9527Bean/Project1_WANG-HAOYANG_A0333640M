# Task2: LASSO with fine lambda grid & beautified plots
# Saves all outputs to your Windows folder.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Save dir ----------
SAVE_DIR = r"C:\Users\13970\Desktop\QF结果图"
os.makedirs(SAVE_DIR, exist_ok=True)

rng = np.random.default_rng(7)  # same seed as Task1

# ---------- Config ----------
T_tr, T_te = 200, 200
c_true = 10
P = c_true * T_tr
b_star = 0.2

cq_grid = [0.5, 0.75, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.25, 1.5, 2, 3, 5, 7, 10]
lambda_grid = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

# ---------- Data ----------
T = T_tr + T_te
S_full = rng.normal(size=(T, P)).astype(np.float64)
S_full = S_full[:, rng.permutation(P)]

beta_raw  = rng.normal(size=P)
beta_star = np.sqrt(b_star) * beta_raw / np.linalg.norm(beta_raw)

eps    = rng.normal(size=T)
R_next = S_full @ beta_star + eps
S_tr, R_tr = S_full[:T_tr], R_next[:T_tr]
S_te, R_te = S_full[T_tr:],  R_next[T_tr:]

# ---------- Lasso (ISTA) ----------
def spectral_norm_power(S, iters=60):
    v = np.random.default_rng(123).normal(size=S.shape[1])
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(iters):
        v = S.T @ (S @ v)
        v /= (np.linalg.norm(v) + 1e-12)
    return float(np.linalg.norm(S @ v))

def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)

def lasso_ista(S, y, lam, max_iter=4000, tol=1e-6):
    Tn, Pn = S.shape
    if lam == 0.0:
        if Pn <= Tn:
            bhat, *_ = np.linalg.lstsq(S, y, rcond=None)
            return bhat
        return S.T @ np.linalg.pinv(S @ S.T) @ y

    sigma_max = spectral_norm_power(S)
    L = (2.0 / Tn) * (sigma_max ** 2)
    eta = 1.0 / (L + 1e-12)

    b = np.zeros(Pn, dtype=np.float64)
    for _ in range(max_iter):
        r = S @ b - y
        g = (2.0 / Tn) * (S.T @ r)
        b_new = soft_threshold(b - eta * g, 2.0 * lam * eta)
        if np.linalg.norm(b_new - b) <= tol * (np.linalg.norm(b) + 1e-12):
            b = b_new
            break
        b = b_new
    return b

def compute_metrics(R_te, Rhat_te, beta_hat):
    Rpi_te  = Rhat_te * R_te
    E_Rpi   = float(np.mean(Rpi_te))
    E_Rpi2  = float(np.mean(Rpi_te**2))
    Sharpe  = E_Rpi / np.sqrt(E_Rpi2) if E_Rpi2 > 0 else np.nan
    denom   = float(np.mean(R_te**2))
    R2      = (2*np.mean(Rhat_te*R_te) - np.mean(Rhat_te**2)) / denom
    beta_n2 = float(np.dot(beta_hat, beta_hat))
    return E_Rpi, E_Rpi2, Sharpe, R2, beta_n2

# ---------- Sweep ----------
rows = []
for cq in cq_grid:
    P1 = min(int(round(cq * T_tr)), P)
    cols = np.arange(P1)
    S1_tr, S1_te = S_tr[:, cols], S_te[:, cols]

    for lam in lambda_grid:
        beta_hat = lasso_ista(S1_tr, R_tr, lam)
        Rhat_te  = S1_te @ beta_hat
        E_Rpi, E_Rpi2, Sharpe, R2, beta_n2 = compute_metrics(R_te, Rhat_te, beta_hat)

        rows.append({"c_q": cq, "P1": P1, "param": lam,
                     "E[R_pi]": E_Rpi, "E[R_pi^2]": E_Rpi2,
                     "Sharpe": Sharpe, "R2_paper": R2, "||beta_hat||^2": beta_n2})

df = pd.DataFrame(rows).sort_values(["param", "c_q"])
df.to_csv(os.path.join(SAVE_DIR, "task2_lasso_metrics_finegrid.csv"), index=False)

# ---------- Pretty plot helper ----------
def pretty_plot(df, xcol, ycol, series_vals, title, ylabel, outname,
                param_name="λ", ylim=None):
    plt.figure(figsize=(9.5, 6.0))
    for v in series_vals:
        sub = df[df["param"] == v]
        plt.plot(sub[xcol], sub[ycol], marker="o", linewidth=1.8, markersize=4,
                 label=f"{param_name}={v}")
    plt.axvline(1.0, linestyle="--", alpha=0.6)
    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel(r"Observed Complexity  $c_q=P_1/T_{tr}$", fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=16, pad=10)

    plt.minorticks_on()
    plt.grid(True, which="major", alpha=0.25)
    plt.grid(True, which="minor", linestyle=":", alpha=0.20)
    plt.tick_params(labelsize=11)

    if ylim is not None:
        y_top = ylim[1]
    else:
        y_top = plt.gca().get_ylim()[1]
    plt.text(1.02, y_top - 0.08*(y_top - plt.gca().get_ylim()[0]),
             "interpolation\n$c_q=1$", fontsize=10)

    plt.legend(ncol=2, fontsize=9, frameon=False, loc="best")
    out = os.path.join(SAVE_DIR, outname)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

# ---------- Make figures (new y-lims) ----------
pretty_plot(df, "c_q", "R2_paper",
            lambda_grid,
            "Out-of-sample $R^2$ vs. Observed Complexity (LASSO)",
            r"$R^2$ (out-of-sample)",
            "T2_fig1_R2_vs_cq_LASSO_fine_v2.png",
            ylim=(-3, 0.5))

pretty_plot(df, "c_q", "E[R_pi]",
            lambda_grid,
            "Expected Timing Return vs. Observed Complexity (LASSO)",
            r"$\mathbb{E}[R^\pi]$",
            "T2_fig2_E_Rpi_vs_cq_LASSO_fine_v2.png")

pretty_plot(df, "c_q", "Sharpe",
            lambda_grid,
            "Sharpe (uncentered) vs. Observed Complexity (LASSO)",
            "Sharpe (uncentered)",
            "T2_fig3_Sharpe_vs_cq_LASSO_fine_v2.png")

pretty_plot(df, "c_q", "||beta_hat||^2",
            lambda_grid,
            r"Parameter Size $\|\hat\beta\|^2$ vs. Observed Complexity (LASSO)",
            r"$\|\hat\beta\|^2$",
            "T2_fig4_beta_norm2_LASSO_fine_v2.png",
            ylim=(-1, 10))
