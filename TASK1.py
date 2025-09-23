# Task1: Ridge (incl. ridgeless) with fine grids & beautified plots
# Saves all outputs to your Windows folder.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Save dir ----------
SAVE_DIR = r"C:\Users\13970\Desktop\QF结果图"
os.makedirs(SAVE_DIR, exist_ok=True)

rng = np.random.default_rng(7)

# ---------- Config ----------
T_tr, T_te = 200, 200
c_true = 10
P = c_true * T_tr
b_star = 0.2

# Fine grids (from your slide)
z_grid  = [0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100]
cq_grid = [0.5, 0.75, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.25, 1.5, 2, 3, 5, 7, 10]

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

# ---------- Ridge estimator ----------
def ridge_beta(S, y, z):
    Tn, Pn = S.shape
    if z == 0:
        if Pn <= Tn:
            bhat, *_ = np.linalg.lstsq(S, y, rcond=None)
            return bhat
        return S.T @ np.linalg.pinv(S @ S.T) @ y
    if Pn <= Tn:
        G = (S.T @ S) / Tn
        rhs = (S.T @ y) / Tn
        return np.linalg.solve(z*np.eye(Pn) + G, rhs)
    A = (S @ S.T) / Tn + z*np.eye(Tn)
    sol = np.linalg.solve(A, y / Tn)
    return S.T @ sol

# ---------- Sweep ----------
rows = []
for cq in cq_grid:
    P1 = min(int(round(cq * T_tr)), P)
    cols = np.arange(P1)
    S1_tr, S1_te = S_tr[:, cols], S_te[:, cols]

    for z in z_grid:
        beta_hat = ridge_beta(S1_tr, R_tr, z)
        Rhat_te  = S1_te @ beta_hat
        Rpi_te   = Rhat_te * R_te

        E_Rpi  = float(np.mean(Rpi_te))
        E_Rpi2 = float(np.mean(Rpi_te**2))
        Sharpe = E_Rpi / np.sqrt(E_Rpi2) if E_Rpi2 > 0 else np.nan
        denom  = float(np.mean(R_te**2))
        R2     = (2*np.mean(Rhat_te*R_te) - np.mean(Rhat_te**2)) / denom
        beta_n2 = float(np.dot(beta_hat, beta_hat))

        rows.append({"c_q": cq, "P1": P1, "param": z,
                     "E[R_pi]": E_Rpi, "E[R_pi^2]": E_Rpi2,
                     "Sharpe": Sharpe, "R2_paper": R2, "||beta_hat||^2": beta_n2})

df = pd.DataFrame(rows).sort_values(["param", "c_q"])
df.to_csv(os.path.join(SAVE_DIR, "task1_metrics_finegrid.csv"), index=False)

# ---------- Pretty plot helper ----------
def pretty_plot(df, xcol, ycol, series_vals, title, ylabel, outname,
                param_name="z", ylim=None):
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

    # 标注插值线
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
            z_grid,
            "Out-of-sample $R^2$ vs. Observed Complexity (Ridge)",
            r"$R^2$ (out-of-sample)",
            "T1_fig1_R2_vs_cq_fine_v2.png",
            ylim=(-3, 0.5))

pretty_plot(df, "c_q", "E[R_pi]",
            z_grid,
            "Expected Timing Return vs. Observed Complexity (Ridge)",
            r"$\mathbb{E}[R^\pi]$",
            "T1_fig2_E_Rpi_vs_cq_fine_v2.png")

pretty_plot(df, "c_q", "Sharpe",
            z_grid,
            "Sharpe (uncentered) vs. Observed Complexity (Ridge)",
            "Sharpe (uncentered)",
            "T1_fig3_Sharpe_vs_cq_fine_v2.png")

pretty_plot(df, "c_q", "||beta_hat||^2",
            z_grid,
            r"Parameter Size $\|\hat\beta\|^2$ vs. Observed Complexity (Ridge)",
            r"$\|\hat\beta\|^2$",
            "T1_fig4_beta_norm2_fine_v2.png",
            ylim=(-1, 7))
