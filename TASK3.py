import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== User config ==================
SAVE_DIR = r"C:\Users\13970\Desktop\QF结果图"
os.makedirs(SAVE_DIR, exist_ok=True)

PAIR_PATHS = {
    "A": {
        "train": r"C:\Users\13970\Desktop\QFProject1\TASK3数据\pairA_train.csv",
        "test":  r"C:\Users\13970\Desktop\QFProject1\TASK3数据\pairA_test_features.csv",
        "out":   os.path.join(SAVE_DIR, "A0333640M_predictions_A.csv"),
    },
    "B": {
        "train": r"C:\Users\13970\Desktop\QFProject1\TASK3数据\pairB_train.csv",
        "test":  r"C:\Users\13970\Desktop\QFProject1\TASK3数据\pairB_test_features.csv",
        "out":   os.path.join(SAVE_DIR, "A0333640M_predictions_B.csv"),
    },
    "C": {
        "train": r"C:\Users\13970\Desktop\QFProject1\TASK3数据\pairC_train.csv",
        "test":  r"C:\Users\13970\Desktop\QFProject1\TASK3数据\pairC_test_features.csv",
        "out":   os.path.join(SAVE_DIR, "A0333640M_predictions_C.csv"),
    },
}

# 超参网格（Task1/2 启发：从弱到中度收缩）
RIDGE_Z_GRID  = [0.0, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
LASSO_L_GRID  = [0.0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1]

# 时间块滚动验证参数（只用训练集）
N_BLOCKS      = 5
MIN_BLOCK_LEN = 50

# 是否生成图
MAKE_PLOTS = True

SEED = 7
rng = np.random.default_rng(SEED)
# =================================================


# ------------------ 基础工具 ------------------
def check_and_align_columns(df_train, df_test):
    if df_train.isna().any().any() or df_test.isna().any().any():
        raise ValueError("NaNs detected.")
    if df_train["t"].duplicated().any() or df_test["t"].duplicated().any():
        raise ValueError("Duplicate t detected.")

    tr_cols = list(df_train.columns)
    te_cols = list(df_test.columns)
    if tr_cols[0] != "t" or tr_cols[-1] != "return":
        raise ValueError("Train must be: t, feature1..P, return")
    if te_cols[0] != "t":
        raise ValueError("Test must be: t, feature1..P")

    feat_train = tr_cols[1:-1]
    feat_test  = te_cols[1:]

    if set(feat_train) != set(feat_test):
        missing = set(feat_train) - set(feat_test)
        extra   = set(feat_test)  - set(feat_train)
        raise ValueError(f"Feature mismatch. Missing in test: {missing}; Extra in test: {extra}")

    df_test = df_test[["t"] + feat_train]

    t_tr = df_train["t"].to_numpy()
    t_te = df_test["t"].to_numpy()
    X_tr = df_train[feat_train].to_numpy(dtype=float)
    y_tr = df_train["return"].to_numpy(dtype=float)
    X_te = df_test[feat_train].to_numpy(dtype=float)
    return X_tr, y_tr, X_te, t_tr, t_te, feat_train

def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0.0] = 1.0
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

def metrics_oos(y_true_next, yhat_next):
    rpi   = yhat_next * y_true_next
    E_Rpi = float(np.mean(rpi))
    denom_uc = float(np.mean(rpi**2))
    SR    = E_Rpi / math.sqrt(denom_uc) if denom_uc > 0 else float("nan")
    denom_R = float(np.mean(y_true_next**2))
    R2    = (2*np.mean(yhat_next*y_true_next) - np.mean(yhat_next**2)) / denom_R
    return E_Rpi, SR, R2


# ------------------ 模型：Ridge & Lasso ------------------
def ridge_fit(X, y, z):
    Tn, Pn = X.shape
    if z == 0.0:
        if Pn > Tn:
            return X.T @ np.linalg.pinv(X @ X.T) @ y
        else:
            return np.linalg.lstsq(X, y, rcond=None)[0]
    G = X.T @ X
    return np.linalg.solve(G + z*np.eye(Pn), X.T @ y)

def spectral_norm_power(X, iters=60):
    Pn = X.shape[1]
    v = rng.normal(size=Pn)
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(iters):
        v = X.T @ (X @ v)
        v /= (np.linalg.norm(v) + 1e-12)
    return np.linalg.norm(X @ v)

def soft_thresh(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)

def lasso_fit_ista(X, y, lam, max_iter=3000, tol=1e-6):
    Tn, Pn = X.shape
    if lam == 0.0:
        if Pn > Tn:
            return X.T @ np.linalg.pinv(X @ X.T) @ y
        else:
            return np.linalg.lstsq(X, y, rcond=None)[0]
    sigma = spectral_norm_power(X, iters=60)
    L = (2.0 / Tn) * (sigma ** 2)
    eta = 1.0 / (L + 1e-12)
    b = np.zeros(Pn)
    for _ in range(max_iter):
        r = X @ b - y
        g = (2.0 / Tn) * (X.T @ r)
        b_new = soft_thresh(b - eta * g, 2.0 * lam * eta)
        if np.linalg.norm(b_new - b) <= tol * (np.linalg.norm(b) + 1e-12):
            b = b_new
            break
        b = b_new
    return b


# ------------------ 时间块滚动验证 & 稳定性 ------------------
def time_blocks_index(T, n_blocks=5, min_block_len=50):
    if T < n_blocks * min_block_len:
        n_blocks = max(2, min(3, T // max(min_block_len, 1)))
    edges = np.linspace(0, T, n_blocks+1, dtype=int)
    blocks = [(int(edges[i]), int(edges[i+1])) for i in range(n_blocks)]
    blocks = [(s, e) for (s, e) in blocks if e > s]
    return blocks

def support(b, thresh=1e-8):
    return set(np.where(np.abs(b) > thresh)[0])

def cosine_sim(u, v):
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu == 0 or nv == 0: return 0.0
    return float(np.dot(u, v) / (nu * nv))

def rolling_cv_family(X, y, family, params_grid):
    Tn = X.shape[0]
    blocks = time_blocks_index(Tn, N_BLOCKS, MIN_BLOCK_LEN)
    results = []
    detail_rows = []
    for p in params_grid:
        SR_list, E_list, R2_list = [], [], []
        coef_list = []
        for i in range(1, len(blocks)):
            train_start, _ = blocks[0][0], blocks[i-1][1]
            valid_start, valid_end = blocks[i]
            X_tr, y_tr = X[train_start:valid_start], y[train_start:valid_start]
            X_vl, y_vl = X[valid_start:valid_end], y[valid_start:valid_end]
            if X_tr.shape[0] < 10 or X_vl.shape[0] < 5:
                continue
            if family == "ridge":
                b = ridge_fit(X_tr, y_tr, p)
            else:
                b = lasso_fit_ista(X_tr, y_tr, p)
            yhat_vl = X_vl @ b
            E, SR, R2 = metrics_oos(y_vl, yhat_vl)
            SR_list.append(SR); E_list.append(E); R2_list.append(R2)
            coef_list.append(b.copy())
            detail_rows.append({
                "family": family, "param": p, "fold": i,
                "E[R_pi]": E, "SR": SR, "R2": R2
            })
        mean_SR = float(np.nanmean(SR_list)) if SR_list else -np.inf
        mean_E  = float(np.nanmean(E_list))  if E_list  else -np.inf
        mean_R2 = float(np.nanmean(R2_list)) if R2_list else -np.inf
        SR_std  = float(np.nanstd(SR_list))  if SR_list else np.inf

        stability = 0.0
        if len(coef_list) >= 2:
            sims = []
            for a in range(len(coef_list)):
                for b_idx in range(a+1, len(coef_list)):
                    if family == "lasso":
                        S1, S2 = support(coef_list[a]), support(coef_list[b_idx])
                        inter = len(S1 & S2); union = len(S1 | S2) or 1
                        sims.append(inter / union)
                    else:
                        sims.append(cosine_sim(coef_list[a], coef_list[b_idx]))
            stability = float(np.mean(sims)) if sims else 0.0

        results.append({
            "family": family, "param": p,
            "mean_SR": mean_SR, "mean_E": mean_E, "mean_R2": mean_R2,
            "SR_std": SR_std, "stability": stability
        })

    results_sorted = sorted(results, key=lambda r: (r["mean_SR"], r["stability"], r["mean_E"], r["mean_R2"]), reverse=True)
    best = results_sorted[0] if results_sorted else None
    return results_sorted, best, detail_rows


# ------------------ 理论先验（稀疏/致密 & 共线性） ------------------
def gini_coeff(x):
    x = np.asarray(x, dtype=float)
    x = np.abs(x)
    s = x.sum()
    if s <= 0: return 0.0
    xs = np.sort(x)
    n = len(xs)
    cum = np.cumsum(xs)
    g = (n + 1 - 2 * np.sum(cum) / s) / n
    return float(g)

def theory_prior_scores(X, y):
    Tn, Pn = X.shape
    y0 = (y - y.mean()) / (y.std(ddof=0) + 1e-12)
    X0 = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-12)
    corr = (X0 * y0[:, None]).mean(axis=0)
    gini_abs_corr = gini_coeff(np.abs(corr))

    if Pn > Tn:
        b0 = X.T @ np.linalg.pinv(X @ X.T) @ y0
    else:
        b0 = np.linalg.lstsq(X0, y0, rcond=None)[0]
    absb = np.abs(b0)
    k = max(1, min(10, int(0.05 * Pn)))
    mass_share_topk = float(np.sort(absb)[::-1][:k].sum() / (absb.sum() + 1e-12))

    if Pn > 200:
        idx = rng.choice(Pn, size=200, replace=False)
        Xs  = X0[:, idx]
    else:
        Xs  = X0
    C = np.corrcoef(Xs.T)
    m = C.shape[0]
    collinearity = float(np.sum(np.abs(C) - np.eye(m)) / (m*(m-1)))
    collinearity = max(0.0, min(1.0, collinearity))

    raw = 0.5 * gini_abs_corr + 0.5 * mass_share_topk - 0.4 * collinearity
    prior_lasso = max(0.0, min(1.0, 0.5 + raw/2.0))
    prior_ridge = 1.0 - prior_lasso

    return {
        "gini_abs_corr": gini_abs_corr,
        "mass_share_topk": mass_share_topk,
        "collinearity": collinearity,
        "prior_lasso": prior_lasso,
        "prior_ridge": prior_ridge
    }


# ------------------ 组合打分（理论 + 证据） ------------------
def choose_model_with_justification(X_tr, y_tr, report_prefix):
    prior = theory_prior_scores(X_tr, y_tr)
    ridge_res, ridge_best, ridge_rows = rolling_cv_family(X_tr, y_tr, "ridge", RIDGE_Z_GRID)
    lasso_res, lasso_best, lasso_rows = rolling_cv_family(X_tr, y_tr, "lasso", LASSO_L_GRID)

    def fam_score(best, prior_w):
        if best is None: return -1e9
        return (0.7 * best["mean_SR"]
                + 0.2 * prior_w
                + 0.1 * best["stability"]
                - 0.05 * best["SR_std"])

    score_ridge = fam_score(ridge_best, prior["prior_ridge"])
    score_lasso = fam_score(lasso_best, prior["prior_lasso"])

    if score_lasso > score_ridge:
        chosen = ("lasso", lasso_best["param"], lasso_best, lasso_res)
    else:
        chosen = ("ridge", ridge_best["param"], ridge_best, ridge_res)

    diag_csv = report_prefix + "_cv_details.csv"
    pd.DataFrame(ridge_rows + lasso_rows).to_csv(diag_csv, index=False, encoding="utf-8")
    summary_csv = report_prefix + "_cv_summary.csv"
    pd.DataFrame(ridge_res + lasso_res).to_csv(summary_csv, index=False, encoding="utf-8")

    rep_path = report_prefix + "_selection_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("Model choice: theory + on-sample evidence\n\n")
        f.write("=== Theory diagnostics ===\n")
        f.write(f"Gini(|corr(y,X_j)|): {prior['gini_abs_corr']:.3f}\n")
        f.write(f"Mass share (top-k |b|min-norm): {prior['mass_share_topk']:.3f}\n")
        f.write(f"Avg |corr| among features: {prior['collinearity']:.3f}\n")
        f.write(f"Prior(Lasso)={prior['prior_lasso']:.3f}, Prior(Ridge)={prior['prior_ridge']:.3f}\n\n")

        def fmt_best(tag, b):
            if b is None:
                return f"{tag}: NA\n"
            return (f"{tag}: param={b['param']}, "
                    f"mean_SR={b['mean_SR']:.4f}, SR_std={b['SR_std']:.4f}, "
                    f"stability={b['stability']:.3f}, mean_E={b['mean_E']:.5f}, mean_R2={b['mean_R2']:.5f}\n")

        f.write("=== Rolling CV (train only) ===\n")
        f.write(fmt_best("Best Ridge", ridge_best))
        f.write(fmt_best("Best Lasso", lasso_best))
        f.write(f"Family score (ridge)={score_ridge:.4f}, (lasso)={score_lasso:.4f}\n\n")

        fam, param, best, _ = chosen
        f.write("=== Final choice ===\n")
        f.write(f"Selected family: {fam.upper()} with param={param}\n")
        f.write("Rationale:\n")
        if fam == "lasso":
            f.write("- Theory prior points to sparsity or ridge CV less stable.\n")
            f.write("- Lasso achieved competitive/better mean Sharpe with higher stability near interpolation.\n")
        else:
            f.write("- Theory prior points to dense/collinear structure or lasso CV less stable.\n")
            f.write("- Ridge aggregated dispersed signals with stronger mean Sharpe or stability.\n")
        f.write("\nArtifacts saved:\n")
        f.write(f"- CV details: {diag_csv}\n- CV summary: {summary_csv}\n")

    return chosen, prior, rep_path, summary_csv, diag_csv


# ------------------ 绘图（CV 证据曲线） ------------------
def plot_family_curves(pair_name, family, results_sorted, out_prefix):
    # 从 results_sorted 提取数组
    params = [r["param"] for r in results_sorted]
    mean_SR = [r["mean_SR"] for r in results_sorted]
    SR_std  = [r["SR_std"]  for r in results_sorted]
    stab    = [r["stability"] for r in results_sorted]
    mean_E  = [r["mean_E"] for r in results_sorted]
    mean_R2 = [r["mean_R2"] for r in results_sorted]

    # 1) Sharpe + 不稳定度阴影（std）
    plt.figure(figsize=(8,5))
    plt.plot(params, mean_SR, marker="o", label="mean Sharpe")
    # 用 std 作为上下界阴影
    low  = [m - s for m, s in zip(mean_SR, SR_std)]
    high = [m + s for m, s in zip(mean_SR, SR_std)]
    plt.fill_between(params, low, high, alpha=0.15, label="Sharpe ± std")
    plt.xlabel(f"{'z' if family=='ridge' else 'lambda'}")
    plt.ylabel("Sharpe (train-CV, uncentered)")
    plt.title(f"Pair {pair_name} {family.capitalize()} — CV Sharpe")
    plt.legend()
    p1 = out_prefix + f"_{family}_cv_sharpe.png"
    plt.tight_layout(); plt.savefig(p1, dpi=160); plt.close()

    # 2) 稳定性曲线
    plt.figure(figsize=(8,5))
    plt.plot(params, stab, marker="o")
    plt.xlabel(f"{'z' if family=='ridge' else 'lambda'}")
    plt.ylabel("Stability (Jaccard for Lasso / Cosine for Ridge)")
    plt.title(f"Pair {pair_name} {family.capitalize()} — CV Stability")
    p2 = out_prefix + f"_{family}_cv_stability.png"
    plt.tight_layout(); plt.savefig(p2, dpi=160); plt.close()

    # 3) E[Rπ] 与 R²（两次单独画）
    plt.figure(figsize=(8,5))
    plt.plot(params, mean_E, marker="o")
    plt.xlabel(f"{'z' if family=='ridge' else 'lambda'}")
    plt.ylabel("E[R_pi] (train-CV)")
    plt.title(f"Pair {pair_name} {family.capitalize()} — CV E[R_pi]")
    p3 = out_prefix + f"_{family}_cv_E.png"
    plt.tight_layout(); plt.savefig(p3, dpi=160); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(params, mean_R2, marker="o")
    plt.xlabel(f"{'z' if family=='ridge' else 'lambda'}")
    plt.ylabel("R2_paper (train-CV)")
    plt.title(f"Pair {pair_name} {family.capitalize()} — CV R2")
    p4 = out_prefix + f"_{family}_cv_R2.png"
    plt.tight_layout(); plt.savefig(p4, dpi=160); plt.close()

    return [p1, p2, p3, p4]


# ------------------ 主流程：每个 pair 训练->选模->预测 ------------------
def process_pair(name, train_path, test_path, out_path):
    print(f"\n=== Processing pair {name} ===")
    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)

    X_tr_raw, y_tr, X_te_raw, t_tr, t_te, feat_cols = check_and_align_columns(df_tr, df_te)

    # 标准化（训练统计量）
    mu, sd = standardize_fit(X_tr_raw)
    X_tr = standardize_apply(X_tr_raw, mu, sd)
    X_te = standardize_apply(X_te_raw, mu, sd)

    # 选择模型（理论 + 训练期证据）
    pref = os.path.join(SAVE_DIR, f"pair_{name}")
    (family, param, best, results_sorted), prior, rep_path, summary_csv, diag_csv = choose_model_with_justification(X_tr, y_tr, pref)

    # 可视化 CV 证据
    if MAKE_PLOTS:
        # 分别画 Ridge 与 Lasso 的曲线
        ridge_res, _, _ = rolling_cv_family(X_tr, y_tr, "ridge", RIDGE_Z_GRID)
        lasso_res, _, _ = rolling_cv_family(X_tr, y_tr, "lasso", LASSO_L_GRID)
        plot_family_curves(name, "ridge", ridge_res, pref)
        plot_family_curves(name, "lasso", lasso_res, pref)

    # 用全训练集拟合并预测
    if family == "ridge":
        b = ridge_fit(X_tr, y_tr, param)
    else:
        b = lasso_fit_ista(X_tr, y_tr, param)
    yhat_te = X_te @ b

    # 输出预测
    out_df = pd.DataFrame({"t": t_te, "yhat": yhat_te})
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved predictions: {out_path}")
    print(f"Report: {rep_path}")
    print(f"CV summary: {summary_csv}")
    print(f"CV details: {diag_csv}")

    # 返回选择结果，汇总到表
    return {
        "pair": name, "family": family, "param": param,
        "mean_SR": best["mean_SR"], "SR_std": best["SR_std"],
        "stability": best["stability"], "mean_E": best["mean_E"], "mean_R2": best["mean_R2"],
        "prior_lasso": prior["prior_lasso"], "prior_ridge": prior["prior_ridge"]
    }


if __name__ == "__main__":
    chosen_rows = []
    for k, v in PAIR_PATHS.items():
        row = process_pair(k, v["train"], v["test"], v["out"])
        chosen_rows.append(row)
    # 汇总所选模型
    summary_path = os.path.join(SAVE_DIR, "chosen_models_summary.csv")
    pd.DataFrame(chosen_rows).to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\nChosen models summary saved: {summary_path}")
    print("All done.")
