"""Problem 3: real-time prediction of flue gas from converter-mouth flame spectra.

Target  : y(t) = c(t + tau(t))  -- the gas that will reach the analyser tau(t) seconds later (Q2 model).
Inputs  : causal features only (spectra and oxygen data up to time t).
Protocol: leave-one-heat-out (LOHO) over all 10 heats; every scaler / PCA / model is fitted on training heats.
Outputs : tables/q3_*.csv, figures/q3_*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from common import (HEATS, HEAT_ID, GROUP, DT, CLEAN, TAB, SEED, WL_GRID, GROUP_COLOR, style, savefig)

style()
STEP = 1.0
GASES = ("co", "co2")
GAS_LABEL = {"co": "CO", "co2": "CO2"}
V_TYPICAL = 3400.0
N_PC = 15


# ---------------------------------------------------------------- data on a 1 s grid (causal features)
def trailing_mean(A, w):
    c = np.cumsum(np.vstack([np.zeros((1,) + A.shape[1:]), A]), axis=0)
    idx = np.arange(1, len(A) + 1)
    lo = np.maximum(idx - w, 0)
    n = (idx - lo).reshape((-1,) + (1,) * (A.ndim - 1))
    return (c[idx] - c[lo]) / n


def load(k):
    sp = np.load(os.path.join(CLEAN, f"heat{k:02d}_spec.npz"))
    pr = np.load(os.path.join(CLEAN, f"heat{k:02d}_proc.npz"))
    dl = np.load(os.path.join(CLEAN, f"heat{k:02d}_delay.npz"))
    t0 = pr["t"]
    tg = np.arange(0, t0[-1] + 1e-9, STEP)
    X = sp["X_lr"].astype(float)
    snv = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    # causal resampling to 1 s: mean of the frames inside (t-1, t]
    bins = np.searchsorted(t0, tg, side="right")
    lo = np.searchsorted(t0, tg - STEP, side="right")
    cs = np.vstack([np.zeros((1, snv.shape[1])), np.cumsum(snv, 0)])
    S1 = (cs[bins] - cs[lo]) / np.maximum(bins - lo, 1)[:, None]
    logT = np.log(np.exp(X).sum(1))
    ct = np.r_[0, np.cumsum(logT)]
    lT = (ct[bins] - ct[lo]) / np.maximum(bins - lo, 1)
    S5 = trailing_mean(S1, 5)                          # 5 s causal smoothing against flicker
    lT5 = trailing_mean(lT[:, None], 5)[:, 0]
    V = np.interp(tg, t0, pr["V"])
    w = int(24 / STEP)
    Qc = np.r_[np.full(w, np.nan), (V[w:] - V[:-w]) / (w * STEP)]  # causal 24 s slope
    Qc[:w] = V[:w] / np.maximum(tg[:w], 1)
    # heat-wise baseline (running mean of the first <= 60 s, causal)
    base = np.array([S5[:max(1, min(i + 1, 60))].mean(0) for i in range(len(tg))])
    tau = np.interp(tg, dl["t"], np.where(np.isnan(dl["tau"]), np.nan, dl["tau"]))
    d = dict(t=tg, S5=S5, base=base, lT5=lT5, V=V, Q=Qc, tau=tau, bps=pr["bps"])
    for g in GASES:
        c = np.interp(tg, t0, pr[g])
        d[g + "_now"] = c
        d[g + "_al"] = np.interp(tg + tau, tg, c, right=np.nan)
    st = np.searchsorted(pr["bps"], V)
    d["stage"] = st
    d["tb"] = np.interp(pr["bps"], V, tg)
    d["mask"] = (tg >= 30) & ~np.isnan(d["co_al"]) & ~np.isnan(d["co2_al"])
    return d


D = {k: load(k) for k in HEATS}
print("rows per heat:", {k: int(D[k]["mask"].sum()) for k in HEATS})


def spec_feats(Z, lT):
    """PC scores + trailing means (10, 20 s) + 10 s slopes, + log total intensity."""
    Z10, Z20 = trailing_mean(Z, 10), trailing_mean(Z, 20)
    dZ = Z - np.vstack([np.repeat(Z[:1], 10, 0), Z[:-10]])
    return np.column_stack([Z, Z10, Z20, dZ, lT])


def proc_feats(d):
    return np.column_stack([d["V"], d["Q"], d["V"] / V_TYPICAL, trailing_mean(d["Q"][:, None], 30)[:, 0]])


def build(train, test, heatnorm=False, n_pc=N_PC):
    """Fit PCA on training heats (masked rows) and return feature dicts {heat: (spec, proc)}."""
    get = (lambda k: D[k]["S5"] - D[k]["base"]) if heatnorm else (lambda k: D[k]["S5"])
    pca = PCA(n_pc, random_state=SEED).fit(np.vstack([get(k)[D[k]["mask"]] for k in train]))
    F = {}
    for k in list(train) + [test]:
        Fs = spec_feats(pca.transform(get(k)), D[k]["lT5"])
        F[k] = (Fs[D[k]["mask"]], proc_feats(D[k])[D[k]["mask"]])
    return F, pca


def Xof(F, k, kind):
    s, p = F[k]
    return {"proc": p, "spec": s, "both": np.column_stack([s, p])}[kind]


def target(k, gas, aligned=True):
    return D[k][gas + ("_al" if aligned else "_now")][D[k]["mask"]]


def lgbm():
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=SEED, verbose=-1)


def fit_predict(model, F, train, test, kind, gas, aligned=True):
    Xtr = np.vstack([Xof(F, k, kind) for k in train])
    ytr = np.concatenate([target(k, gas, aligned) for k in train])
    if isinstance(model, (Ridge, PLSRegression)):
        sc = StandardScaler().fit(Xtr)
        model.fit(sc.transform(Xtr), ytr)
        return np.ravel(model.predict(sc.transform(Xof(F, test, kind)))), model
    model.fit(Xtr, ytr)
    return model.predict(Xof(F, test, kind)), model


def metrics(y, p, d=None):
    out = dict(R2=1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), RMSE=np.sqrt(np.mean((y - p) ** 2)),
               MAE=np.mean(np.abs(y - p)))
    if d is not None:
        t = d["t"][d["mask"]]
        act = (t > 30) & (t < t[-1] - 30) & (y > 2)
        out["MAPE_active"] = np.mean(np.abs(y[act] - p[act]) / y[act]) * 100
    return out


# ---------------------------------------------------------------- LOHO main loop
CONFIGS = {
    "M0 process only": dict(kind="proc", model="lgbm", aligned=True),
    "M1 spectra only": dict(kind="spec", model="lgbm", aligned=True),
    "M2 spectra + process (no delay alignment)": dict(kind="both", model="lgbm", aligned=False),
    "M3 spectra + process + delay alignment": dict(kind="both", model="lgbm", aligned=True),
    "M4 = M3 + heat-wise normalization": dict(kind="both", model="lgbm", aligned=True, heatnorm=True),
    "PLS (M3 features)": dict(kind="both", model="pls", aligned=True),
    "Ridge (M3 features)": dict(kind="both", model="ridge", aligned=True),
}
PRED = {c: {g: {} for g in GASES} for c in CONFIGS}
PRED.update({c: {g: {} for g in GASES} for c in ("Grouped (6B / 6C)", "Ensemble: equal weights",
                                                  "Ensemble: similarity weights")})
imp_rows = []
for h in HEATS:
    train = [k for k in HEATS if k != h]
    F, _ = build(train, h)
    Fn, _ = build(train, h, heatnorm=True)
    for cname, cfg in CONFIGS.items():
        FF = Fn if cfg.get("heatnorm") else F
        for g in GASES:
            m = {"lgbm": lgbm(), "pls": PLSRegression(8), "ridge": Ridge(10.0)}[cfg["model"]]
            p, mdl = fit_predict(m, FF, train, h, cfg["kind"], g, cfg["aligned"])
            PRED[cname][g][h] = p
            if cname.startswith("M3"):
                imp_rows.append(pd.Series(mdl.booster_.feature_importance("gain"), name=(h, g)))
    # grouped model: train only on heats of the same group
    same = [k for k in train if GROUP[k] == GROUP[h]]
    Fg, _ = build(same, h)
    for g in GASES:
        PRED["Grouped (6B / 6C)"][g][h] = fit_predict(lgbm(), Fg, same, h, "both", g)[0]
    # per-heat models -> equal / similarity-weighted ensemble
    sims = {}
    b_h = D[h]["S5"][(D[h]["t"] >= 0) & (D[h]["t"] < 60)].mean(0)
    for k in train:
        b_k = D[k]["S5"][(D[k]["t"] >= 0) & (D[k]["t"] < 60)].mean(0)
        sims[k] = np.sum((b_h - b_k) ** 2)
    sig2 = np.median(list(sims.values()))
    wts = np.array([np.exp(-sims[k] / sig2) for k in train]); wts /= wts.sum()
    for g in GASES:
        preds = np.array([fit_predict(lgbm(), F, [k], h, "both", g)[0] for k in train])
        PRED["Ensemble: equal weights"][g][h] = preds.mean(0)
        PRED["Ensemble: similarity weights"][g][h] = wts @ preds
    print("fold", h, "done", flush=True)

# ---------------------------------------------------------------- metrics tables
rows = []
for cname in PRED:
    for g in GASES:
        for h in HEATS:
            y = target(h, g, True)
            rows.append(dict(config=cname, gas=GAS_LABEL[g], heat=h, group=GROUP[h],
                             **metrics(y, PRED[cname][g][h], D[h])))
met = pd.DataFrame(rows)
met.to_csv(os.path.join(TAB, "q3_loho_metrics.csv"), index=False)
summ = met.groupby(["config", "gas"]).agg(R2_median=("R2", "median"), R2_mean=("R2", "mean"), R2_min=("R2", "min"),
                                          RMSE_mean=("RMSE", "mean"), MAE_mean=("MAE", "mean"),
                                          MAPE_active_mean=("MAPE_active", "mean")).reset_index()
summ.to_csv(os.path.join(TAB, "q3_summary_all_configs.csv"), index=False)
abl = summ[summ.config.str.match(r"M\d")]
abl.to_csv(os.path.join(TAB, "q3_ablation.csv"), index=False)
comb = summ[summ.config.isin(["M3 spectra + process + delay alignment", "Grouped (6B / 6C)",
                               "Ensemble: equal weights", "Ensemble: similarity weights"])]
comb.to_csv(os.path.join(TAB, "q3_combination.csv"), index=False)
pd.set_option("display.width", 200)
print(summ.round(3).to_string())

# unaligned task for reference (predict the currently measured gas)
ref = []
for g in GASES:
    for h in HEATS:
        y = target(h, g, False)
        ref.append(dict(gas=GAS_LABEL[g], heat=h, **metrics(y, PRED["M2 spectra + process (no delay alignment)"][g][h])))
pd.DataFrame(ref).to_csv(os.path.join(TAB, "q3_unaligned_task_reference.csv"), index=False)

# choose final model: best mean R2 over both gases among combination candidates and M3/M4
cand = summ[summ.config.isin(["M3 spectra + process + delay alignment", "M4 = M3 + heat-wise normalization",
                               "Grouped (6B / 6C)", "Ensemble: equal weights", "Ensemble: similarity weights"])]
FINAL = cand.groupby("config")["R2_mean"].mean().idxmax()
print("FINAL model:", FINAL)

# ---------------------------------------------------------------- stage-wise and transition errors
stage_rows, trans_rows = [], []
for g in GASES:
    for h in HEATS:
        d = D[h]; y = target(h, g); t = d["t"][d["mask"]]; st = d["stage"][d["mask"]]
        for cname in ("M0 process only", FINAL):
            p = PRED[cname][g][h]
            for s in range(4):
                m = st == s
                if m.sum() > 10:
                    stage_rows.append(dict(gas=GAS_LABEL[g], heat=h, config=cname, stage=s + 1,
                                           RMSE=np.sqrt(np.mean((y[m] - p[m]) ** 2))))
            near = np.zeros_like(t, bool)
            for tb in d["tb"]:
                near |= np.abs(t - tb) <= 30
            dq = np.abs(np.gradient(d["Q"][d["mask"]]))
            jump = dq >= np.quantile(dq, 0.9)
            for nm, m in (("Within +/-30 s of stage breakpoints", near), ("Top 10% |dQ/dt|", jump),
                          ("All other times", ~near & ~jump)):
                trans_rows.append(dict(gas=GAS_LABEL[g], heat=h, config=cname, region=nm,
                                       RMSE=np.sqrt(np.mean((y[m] - p[m]) ** 2))))
sdf = pd.DataFrame(stage_rows)
sdf.groupby(["gas", "config", "stage"])["RMSE"].mean().unstack().to_csv(os.path.join(TAB, "q3_stagewise_rmse.csv"))
tdf = pd.DataFrame(trans_rows)
tsum = tdf.groupby(["gas", "region", "config"])["RMSE"].mean().unstack()
tsum.to_csv(os.path.join(TAB, "q3_transition_rmse.csv"))
print(tsum.round(3))

# ---------------------------------------------------------------- conformal 90% intervals (residuals of other heats)
cov_rows = []
INT = {g: {} for g in GASES}
for g in GASES:
    res = {h: target(h, g) - PRED[FINAL][g][h] for h in HEATS}
    for h in HEATS:
        others = [k for k in HEATS if k != h]
        st_h = D[h]["stage"][D[h]["mask"]]
        q = np.empty_like(st_h, dtype=float)
        for s in range(4):
            pool = np.concatenate([np.abs(res[k][D[k]["stage"][D[k]["mask"]] == s]) for k in others])
            n = len(pool)
            q[st_h == s] = np.quantile(pool, min(1, np.ceil((n + 1) * 0.9) / n)) if n else np.nan
        INT[g][h] = q
        cov_rows.append(dict(gas=GAS_LABEL[g], heat=h, coverage=np.mean(np.abs(res[h]) <= q),
                             mean_halfwidth=np.mean(q)))
cov = pd.DataFrame(cov_rows)
cov.to_csv(os.path.join(TAB, "q3_conformal_coverage.csv"), index=False)
print(cov.groupby("gas")[["coverage", "mean_halfwidth"]].mean())

# ---------------------------------------------------------------- online correction with delayed measurements
def online(h, g, alpha, N):
    """At time t, the analyser has delivered measurements up to t; a measurement taken at s reveals the
    true aligned target for bath time u with u + tau(u) = s. Probe works every N seconds."""
    d = D[h]; m = d["mask"]; t = d["t"][m]; y = target(h, g); p = PRED[FINAL][g][h].copy()
    arrival = t + d["tau"][m]                       # when y(u) becomes known
    if N is None:
        return p
    probe_times = np.arange(0, t[-1] + d["tau"][m].max() + N, N)
    # for each probe time, the newest u whose measurement has arrived
    known = np.searchsorted(arrival, probe_times, side="right") - 1
    b, out, j = 0.0, np.empty_like(p), 0
    events = [(pt, kn) for pt, kn in zip(probe_times, known) if kn >= 0]
    last = -1
    for i, ti in enumerate(t):
        while j < len(events) and events[j][0] <= ti:
            kn = events[j][1]
            if kn > last:
                b = (1 - alpha) * b + alpha * (y[kn] - p[kn])
                last = kn
            j += 1
        out[i] = p[i] + b
    return out


on_rows = []
for alpha in (0.05, 0.1, 0.2, 0.3, 0.5):
    for N in (1, 5, 10, 30, 60, 120, None):
        for g in GASES:
            for h in HEATS:
                y = target(h, g)
                pc = online(h, g, alpha, N)
                on_rows.append(dict(alpha=alpha, probe_interval_s=("none" if N is None else N), gas=GAS_LABEL[g],
                                    heat=h, RMSE=np.sqrt(np.mean((y - pc) ** 2)),
                                    R2=1 - np.sum((y - pc) ** 2) / np.sum((y - y.mean()) ** 2)))
on = pd.DataFrame(on_rows)
on.to_csv(os.path.join(TAB, "q3_online_correction_all.csv"), index=False)
onsum = on.groupby(["alpha", "probe_interval_s", "gas"], sort=False)[["RMSE", "R2"]].mean().reset_index()
onsum.to_csv(os.path.join(TAB, "q3_online_correction_summary.csv"), index=False)
best_alpha = onsum[onsum.probe_interval_s == 1].groupby("alpha")["RMSE"].mean().idxmin()
print("online correction, alpha =", best_alpha)
print(onsum[onsum.alpha == best_alpha].round(3).to_string())

# ---------------------------------------------------------------- interpretability
# PLS VIP on wavelengths (pooled, aligned CO), and LightGBM gain importance
Xall = np.vstack([D[k]["S5"][D[k]["mask"]] for k in HEATS])
vip_tab = {}
for g in GASES:
    yall = np.concatenate([target(k, g) for k in HEATS])
    pls = PLSRegression(8).fit(StandardScaler().fit_transform(Xall), yall)
    T, W, Qp = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    ss = np.sum(T ** 2, 0) * Qp[0] ** 2
    vip = np.sqrt(Xall.shape[1] * (W ** 2 / np.sum(W ** 2, 0)) @ ss / ss.sum())
    vip_tab[GAS_LABEL[g]] = vip
pd.DataFrame(dict(wavelength_nm=WL_GRID, **vip_tab)).to_csv(os.path.join(TAB, "q3_vip_wavelength.csv"), index=False)
fnames = ([f"PC{i + 1}" for i in range(N_PC)] + [f"PC{i + 1}_mean10s" for i in range(N_PC)] +
          [f"PC{i + 1}_mean20s" for i in range(N_PC)] + [f"PC{i + 1}_slope10s" for i in range(N_PC)] +
          ["logTotalIntensity", "CumO2", "LanceFlow", "Progress", "LanceFlow_mean30s"])
imp = pd.concat(imp_rows, axis=1)
imp.index = fnames
impm = pd.DataFrame({GAS_LABEL[g]: imp.loc[:, [c for c in imp.columns if c[1] == g]].mean(1) for g in GASES})
impm = impm / impm.sum()
impm.sort_values("CO", ascending=False).to_csv(os.path.join(TAB, "q3_feature_importance.csv"))
spec_share = impm.drop(["CumO2", "LanceFlow", "Progress", "LanceFlow_mean30s"]).sum()
print("share of gain from spectral features:", spec_share.round(3).to_dict())

# heat clustering (heat-level descriptors)
st_tab = pd.read_csv(os.path.join(TAB, "q2_stages.csv"))
desc = []
pca_all = PCA(3, random_state=SEED).fit(Xall)
for k in HEATS:
    d = D[k]
    desc.append(np.r_[pca_all.transform(d["S5"][d["mask"]]).mean(0), d["t"][-1] / 60, d["co_now"].max(),
                      st_tab.loc[st_tab.heat == k, ["Vfrac_break1", "Vfrac_break2"]].values.ravel(),
                      np.nanmean(d["tau"])])
desc = StandardScaler().fit_transform(np.array(desc))
Z = linkage(desc, "ward")

# ---------------------------------------------------------------- figures
def pred_plot(ax, h, g, show_int=True):
    d = D[h]; t = d["t"][d["mask"]] / 60; y = target(h, g); p = PRED[FINAL][g][h]
    if show_int:
        ax.fill_between(t, np.clip(p - INT[g][h], 0, None), p + INT[g][h], color="C0", alpha=0.18, lw=0,
                        label="90% conformal interval")
    ax.plot(t, y, color="k", lw=1.0, label="Measured (delay-aligned)")
    ax.plot(t, p, color="C0", lw=1.1, label="Predicted from flame spectra")
    for tb in d["tb"]:
        ax.axvline(tb / 60, color="0.5", ls=":", lw=0.8)


for h in HEATS:
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, g in zip(axs, GASES):
        pred_plot(ax, h, g)
        r = met[(met.config == FINAL) & (met.gas == GAS_LABEL[g]) & (met.heat == h)].iloc[0]
        ax.set_ylabel(GAS_LABEL[g]); ax.set_title(f"{GAS_LABEL[g]}: R2 = {r.R2:.3f}, RMSE = {r.RMSE:.2f}")
    axs[0].legend(fontsize=7); axs[1].set_xlabel("Time (min)")
    fig.suptitle(f"Heat {h} ({HEAT_ID[h]}): leave-one-heat-out real-time prediction ({FINAL})", fontsize=10)
    fig.tight_layout()
    savefig(fig, f"q3_pred_heat{h:02d}")

for g in GASES:
    fig, axs = plt.subplots(2, 5, figsize=(20, 7), sharey=True)
    for ax, h in zip(axs.flat, HEATS):
        pred_plot(ax, h, g)
        r = met[(met.config == FINAL) & (met.gas == GAS_LABEL[g]) & (met.heat == h)].iloc[0]
        ax.set_title(f"Heat {h}: R2 = {r.R2:.2f}")
        ax.set_xlabel("Time (min)")
    axs[0, 0].set_ylabel(GAS_LABEL[g]); axs[1, 0].set_ylabel(GAS_LABEL[g]); axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"Leave-one-heat-out prediction of {GAS_LABEL[g]} for all ten heats ({FINAL})")
    fig.tight_layout()
    savefig(fig, f"q3_pred_grid_{g}")

fig, axs = plt.subplots(1, 2, figsize=(11, 5))
for ax, g in zip(axs, GASES):
    for h in HEATS:
        ax.scatter(target(h, g), PRED[FINAL][g][h], s=2, alpha=0.3, color=GROUP_COLOR[GROUP[h]])
    lim = [0, max(target(h, g).max() for h in HEATS) * 1.05]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"Measured {GAS_LABEL[g]}"); ax.set_ylabel(f"Predicted {GAS_LABEL[g]}")
    ax.set_title(f"{GAS_LABEL[g]} (blue: 6B heats, red: 6C heats)")
savefig(fig, "q3_scatter")

fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
order = [c for c in PRED]
for ax, g in zip(axs, GASES):
    q = met[met.gas == GAS_LABEL[g]].groupby("config")["R2"].quantile([0.25, 0.5, 0.75]).unstack().loc[order]
    cols = ["C7" if c.startswith(("PLS", "Ridge")) else ("C1" if c.startswith(("Grouped", "Ensemble")) else "C0")
            for c in order]
    ax.barh(range(len(order)), q[0.5], color=cols)
    ax.errorbar(q[0.5], range(len(order)), xerr=[q[0.5] - q[0.25], q[0.75] - q[0.5]], fmt="none", ecolor="k", lw=0.9)
    for i, v in enumerate(q[0.5]):
        ax.text(max(v, -0.45) + 0.02, i, f"{v:.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order if g == "co" else [], fontsize=8)
    ax.set_xlim(-0.5, 1.05); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Median LOHO R2 over 10 heats (whisker: interquartile range)")
    ax.set_title(GAS_LABEL[g])
fig.suptitle("Ablation and model-combination comparison (10-fold leave-one-heat-out)")
fig.tight_layout()
savefig(fig, "q3_ablation_bar")

# transition zoom: heat with the largest M0 - final gap near breakpoints
gap = tdf[(tdf.region.str.startswith("Within")) & (tdf.gas == "CO")].pivot(index="heat", columns="config", values="RMSE")
hz = int((gap["M0 process only"] - gap[FINAL]).idxmax())
d = D[hz]; t = d["t"][d["mask"]] / 60; tb = d["tb"][0] / 60
fig, ax = plt.subplots(figsize=(8, 4))
m = (t > tb - 1.5) & (t < tb + 2.5)
ax.plot(t[m], target(hz, "co")[m], "k", label="Measured (delay-aligned)")
ax.plot(t[m], PRED["M0 process only"]["co"][hz][m], "C1--", label="M0 process only")
ax.plot(t[m], PRED[FINAL]["co"][hz][m], "C0", label="Final model (with spectra)")
ax.axvline(tb, color="0.5", ls=":"); ax.set_xlabel("Time (min)"); ax.set_ylabel("CO")
ax.set_title(f"Heat {hz}: CO around the start of decarburization ramp-up"); ax.legend()
savefig(fig, "q3_zoom_transition")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
sub = onsum[onsum.alpha == best_alpha]
for ax, g in zip(axs, GASES):
    s2 = sub[sub.gas == GAS_LABEL[g]]
    xs = [v for v in s2.probe_interval_s if v != "none"]
    ax.plot(xs, s2[s2.probe_interval_s != "none"]["RMSE"], "o-", label="With online bias correction")
    ax.axhline(float(s2[s2.probe_interval_s == "none"]["RMSE"].iloc[0]), color="C3", ls="--",
               label="Spectra only, no probe")
    ax.set_xscale("log"); ax.set_xlabel("Probe sampling interval N (s)"); ax.set_ylabel("Mean RMSE")
    ax.set_title(f"{GAS_LABEL[g]} (alpha = {best_alpha})"); ax.legend()
fig.suptitle("Reducing probe usage: prediction error versus probe sampling interval")
savefig(fig, "q3_sampling_interval")

fig, ax = plt.subplots(figsize=(9, 3.8))
for g in GASES:
    ax.plot(WL_GRID, vip_tab[GAS_LABEL[g]], label=GAS_LABEL[g])
ax.axhline(1, color="k", ls="--", lw=0.8, label="VIP = 1")
pk = pd.read_csv(os.path.join(TAB, "q1_emission_peaks.csv")).head(6)
for wv in pk.wavelength_nm:
    ax.axvline(wv, color="0.6", ls=":", lw=0.8)
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("PLS VIP score"); ax.legend()
ax.set_title("Wavelength importance for gas prediction (dotted: main emission peaks)")
savefig(fig, "q3_vip_wavelength")

fig, ax = plt.subplots(figsize=(7, 4))
dendrogram(Z, labels=[f"H{k} ({GROUP[k]})" for k in HEATS], ax=ax, color_threshold=0.7 * Z[:, 2].max())
ax.set_ylabel("Ward distance"); ax.set_title("Hierarchical clustering of heats (heat-level descriptors)")
ax.grid(False)
savefig(fig, "q3_cluster_dendrogram")

fig, ax = plt.subplots(figsize=(7, 5))
top = impm.sort_values("CO", ascending=False).head(15)[::-1]
ax.barh(top.index, top["CO"], color=["C2" if n in ("CumO2", "LanceFlow", "Progress", "LanceFlow_mean30s") else "C0"
                                      for n in top.index])
ax.set_xlabel("Share of total gain"); ax.set_title("LightGBM feature importance for CO (green: process features)")
savefig(fig, "q3_feature_importance")
pd.Series(dict(final_model=FINAL, best_alpha=best_alpha, zoom_heat=hz)).to_csv(os.path.join(TAB, "q3_choices.csv"))
print("Q3 finished")
