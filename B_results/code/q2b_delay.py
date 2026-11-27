"""Problem 2 (part B): flue-gas time delay.

Transport model (plug flow in the flue duct, gas flow proportional to lance flow):
    V(t + tau_K(t)) - V(t) = K,   tau(t) = tau0 + tau_K(t)   =>  steady state tau = tau0 + K / Q
Parameters are chosen by a predictability criterion: leave-one-heat-out R^2 of a ridge model that
predicts the delay-aligned gas signal from real-time spectral features. Uncertainty by heat bootstrap.
Outputs: tables/q2_delay_*.csv, figures/q2_delay_*.png, data_clean/heatXX_delay.npz
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import theilslopes, spearmanr
from sklearn.decomposition import PCA
from common import (HEATS, HEAT_ID, GROUP, DT, CLEAN, TAB, SEED, GROUP_COLOR, load_clean, style, savefig)

style()
STEP = 1.0                  # analysis resolution (s)
T_START, T_TAIL = 60.0, 180.0
rng = np.random.default_rng(SEED)


def load_heat(k, dt_override=None):
    sp = np.load(os.path.join(CLEAN, f"heat{k:02d}_spec.npz"))
    pr = np.load(os.path.join(CLEAN, f"heat{k:02d}_proc.npz"))
    t = pr["t"] if dt_override is None else np.arange(len(pr["t"])) * dt_override
    tg = np.arange(0, t[-1] + 1e-9, STEP)
    X = sp["X_lr"].astype(float)          # cleaned per-frame spectra (outliers removed, flicker kept)
    Xs = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)        # SNV per frame
    Xg = np.array([np.interp(tg, t, Xs[:, j]) for j in range(Xs.shape[1])]).T
    logT = np.interp(tg, t, np.log(np.exp(X).sum(1)))
    # same 5-s trailing mean as the Q3 features (suppresses frame-to-frame flame flicker)
    k5 = int(5 / STEP)
    cs = np.vstack([np.zeros((1, Xg.shape[1])), np.cumsum(Xg, 0)])
    idx = np.arange(1, len(tg) + 1); lo = np.maximum(idx - k5, 0)
    Xg = (cs[idx] - cs[lo]) / (idx - lo)[:, None]
    cl = np.r_[0, np.cumsum(logT)]
    logT = (cl[idx] - cl[lo]) / (idx - lo)
    V = np.interp(tg, t, pr["V"]); Q = np.interp(tg, t, pr["Q"])
    return dict(t=tg, Xs=Xg, logT=logT, V=V, Q=Q, co=np.interp(tg, t, pr["co"]),
                co2=np.interp(tg, t, pr["co2"]), bps=pr["bps"])


D = {k: load_heat(k) for k in HEATS}


def tau_of(d, tau0, K):
    t, V = d["t"], d["V"] + 1e-6 * d["t"]
    tk = np.interp(V + K, V, t, right=np.nan) - t if K > 0 else np.zeros_like(t)
    return tau0 + tk


def shifted(d, y, tau):
    return np.interp(d["t"] + tau, d["t"], y, right=np.nan)


def eval_mask(d):
    return (d["t"] >= T_START) & (d["t"] <= d["t"][-1] - T_TAIL)


def fold_features(n_pc=10):
    """Per LOHO fold: PCA on training heats' SNV spectra; features = PCs, 2-s backward slope, logT, Q."""
    F = {}
    for h in HEATS:
        tr = [k for k in HEATS if k != h]
        pca = PCA(n_pc, random_state=SEED).fit(np.vstack([D[k]["Xs"][eval_mask(D[k])] for k in tr]))
        feats = {}
        for k in HEATS:
            Z = pca.transform(D[k]["Xs"])
            dZ = Z - np.vstack([Z[:2], Z[:-2]])
            feats[k] = np.column_stack([Z, dZ, D[k]["logT"], D[k]["Q"]])[eval_mask(D[k])]
        F[h] = feats
    return F


def ridge_solvers(F, alpha=10.0):
    S = {}
    for h in HEATS:
        tr = [k for k in HEATS if k != h]
        Xtr = np.vstack([F[h][k] for k in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
        A = Z.T @ Z + alpha * np.diag(np.r_[0, np.ones(Z.shape[1] - 1)])
        W = np.linalg.solve(A, Z.T)
        Zte = np.column_stack([np.ones(len(F[h][h])), (F[h][h] - mu) / sd])
        S[h] = (W, Zte, tr)
    return S


def loho_r2(S, targets):
    """targets: dict heat -> aligned target on eval rows. Returns array of per-heat test R^2."""
    out = np.empty(len(HEATS))
    for i, h in enumerate(HEATS):
        W, Zte, tr = S[h]
        ytr = np.concatenate([targets[k] for k in tr])
        if np.isnan(ytr).any() or np.isnan(targets[h]).any():
            out[i] = np.nan; continue
        yp = Zte @ (W @ ytr)
        yt = targets[h]
        out[i] = 1 - np.sum((yp - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)
    return out


def targets_for(gas, tau0=0.0, K=0.0, L=None):
    out = {}
    for k in HEATS:
        d = D[k]
        tau = np.full_like(d["t"], L) if L is not None else tau_of(d, tau0, K)
        out[k] = shifted(d, d[gas], tau)[eval_mask(d)]
    return out


F = fold_features(10)
S = ridge_solvers(F)

K_GRID = np.arange(0, 410, 10.0)
T0_GRID = np.arange(0, 32, 2.0)
L_GRID = np.arange(0, 101, 1.0)
res = {}
for gas in ("co", "co2"):
    R2_2d = np.empty((len(T0_GRID), len(K_GRID), len(HEATS)))
    for i, t0 in enumerate(T0_GRID):
        for j, K in enumerate(K_GRID):
            R2_2d[i, j] = loho_r2(S, targets_for(gas, t0, K))
    R2_c = np.array([loho_r2(S, targets_for(gas, L=L)) for L in L_GRID])
    res[gas] = (R2_2d, R2_c)
    print(gas, "done")

# Timing marker: CO. CO is the direct decarburization product with high signal-to-noise; CO2 is dominated by
# the slow stage-wise trend (high early, low later), so its alignment criterion is flat and biased towards long lags.
# The CO + CO2 combined and CO2-only optima are still reported for comparison.
R2_2d = res["co"][0]
R2_c = res["co"][1]
R2_2d_comb = (res["co"][0] + res["co2"][0]) / 2
R2_c_comb = (res["co"][1] + res["co2"][1]) / 2
crit2 = R2_2d.mean(-1); critc = R2_c.mean(-1)
i0, j0 = np.unravel_index(np.nanargmax(crit2), crit2.shape)
tau0_hat, K_hat = T0_GRID[i0], K_GRID[j0]
jk = np.nanargmax(crit2[0]); K1_hat = K_GRID[jk]                    # tau0 = 0 model
L_hat = L_GRID[np.nanargmax(critc)]
rows = [dict(model="No delay", params="-", n_params=0, cv_R2=critc[0],
             cv_R2_CO=res["co"][1][0].mean(), cv_R2_CO2=res["co2"][1][0].mean()),
        dict(model="Constant delay", params=f"L = {L_hat:.0f} s", n_params=1, cv_R2=critc.max(),
             cv_R2_CO=res["co"][1][np.argmax(critc)].mean(), cv_R2_CO2=res["co2"][1][np.argmax(critc)].mean()),
        dict(model="Transport, tau0 = 0", params=f"K = {K1_hat:.0f} m3", n_params=1, cv_R2=crit2[0].max(),
             cv_R2_CO=res["co"][0][0, jk].mean(), cv_R2_CO2=res["co2"][0][0, jk].mean()),
        dict(model="Transport, tau0 + K", params=f"tau0 = {tau0_hat:.0f} s, K = {K_hat:.0f} m3", n_params=2,
             cv_R2=crit2.max(), cv_R2_CO=res["co"][0][i0, j0].mean(), cv_R2_CO2=res["co2"][0][i0, j0].mean())]
c2c = R2_2d_comb.mean(-1); a, b = np.unravel_index(np.nanargmax(c2c), c2c.shape)
rows.append(dict(model="[CO + CO2 combined] transport optimum", params=f"tau0 = {T0_GRID[a]:.0f} s, K = {K_GRID[b]:.0f} m3",
                 n_params=2, cv_R2=c2c.max()))
rows.append(dict(model="[CO + CO2 combined] constant optimum", params=f"L = {L_GRID[np.argmax(R2_c_comb.mean(-1))]:.0f} s",
                 n_params=1, cv_R2=R2_c_comb.mean(-1).max()))
# per-gas optima
for gas in ("co2",):
    c2 = res[gas][0].mean(-1); a, b = np.unravel_index(np.nanargmax(c2), c2.shape)
    cc = res[gas][1].mean(-1)
    rows.append(dict(model=f"[{gas.upper()} only] transport optimum", params=f"tau0 = {T0_GRID[a]:.0f} s, K = {K_GRID[b]:.0f} m3",
                     n_params=2, cv_R2=c2.max()))
    rows.append(dict(model=f"[{gas.upper()} only] constant optimum", params=f"L = {L_GRID[np.argmax(cc)]:.0f} s",
                     n_params=1, cv_R2=cc.max()))
cmp_df = pd.DataFrame(rows)

# paired heat-level comparison transport vs constant
d_heat = R2_2d[i0, j0] - R2_c[np.argmax(critc)]
cmp_df.to_csv(os.path.join(TAB, "q2_delay_model_comparison.csv"), index=False)
print(cmp_df)
print("transport - constant per heat:", np.round(d_heat, 4))

# ---------------------------------------------------------------- bootstrap over heats
B = 2000
boot = []
for b in range(B):
    idx = rng.integers(0, len(HEATS), len(HEATS))
    c2 = R2_2d[:, :, idx].mean(-1); a, bb = np.unravel_index(np.nanargmax(c2), c2.shape)
    cc = R2_c[:, idx].mean(-1)
    boot.append((T0_GRID[a], K_GRID[bb], L_GRID[np.argmax(cc)],
                 (c2[a, bb] - cc.max())))
boot = np.array(boot)
ci = lambda x: np.percentile(x, [2.5, 97.5])
par = pd.DataFrame([
    dict(parameter="tau0 (s)", estimate=tau0_hat, ci_low=ci(boot[:, 0])[0], ci_high=ci(boot[:, 0])[1]),
    dict(parameter="K (m3)", estimate=K_hat, ci_low=ci(boot[:, 1])[0], ci_high=ci(boot[:, 1])[1]),
    dict(parameter="Constant delay L (s)", estimate=L_hat, ci_low=ci(boot[:, 2])[0], ci_high=ci(boot[:, 2])[1]),
    dict(parameter="CV R2 gain transport vs constant", estimate=crit2.max() - critc.max(),
         ci_low=ci(boot[:, 3])[0], ci_high=ci(boot[:, 3])[1]),
    dict(parameter="P(transport better than constant)", estimate=np.mean(boot[:, 3] > 0), ci_low=np.nan, ci_high=np.nan),
])
par.to_csv(os.path.join(TAB, "q2_delay_params.csv"), index=False)
print(par)

# Adopted model: tau = K / Q (tau0 = 0). The 2-parameter fit gives tau0 close to 0 and tau0, K are not separately
# identifiable (flow varies little within a blow), so the 1-parameter transport model is used.
boot1 = np.array([K_GRID[np.nanargmax(R2_2d[0][:, rng.integers(0, len(HEATS), len(HEATS))].mean(-1))]
                  for _ in range(B)])
TAU0_USE, K_USE = 0.0, K1_hat
par = pd.concat([par, pd.DataFrame([dict(parameter="ADOPTED K (m3), model tau = K/Q", estimate=K_USE,
                                         ci_low=np.percentile(boot1, 2.5), ci_high=np.percentile(boot1, 97.5))])])
par.to_csv(os.path.join(TAB, "q2_delay_params.csv"), index=False)
print(par)
QS = np.arange(3.0, 7.01, 0.5)
tq = boot1[:, None] / QS[None, :]
flow_tab = pd.DataFrame(dict(Q_m3_per_s=QS, tau_s=K_USE / QS,
                             ci_low=np.percentile(tq, 2.5, 0), ci_high=np.percentile(tq, 97.5, 0)))
flow_tab.to_csv(os.path.join(TAB, "q2_delay_by_flow.csv"), index=False)
print(flow_tab.round(1))

# ---------------------------------------------------------------- auxiliary methods
# B: windowed GCC-PHAT between d/dt PC1 (global PCA) and d/dt CO
pca_all = PCA(5, random_state=SEED).fit(np.vstack([D[k]["Xs"] for k in HEATS]))
win_rows = []
for k in HEATS:
    d = D[k]
    z = pca_all.transform(d["Xs"])[:, 0]
    sgn = np.sign(np.corrcoef(z, d["co"])[0, 1]) or 1
    x = np.gradient(sgn * z); y = np.gradient(d["co"])
    W, STP, LMAX = 120, 20, 90
    for s0 in range(int(T_START), int(len(d["t"]) - W - LMAX), STP):
        xs = x[s0:s0 + W]; ys = y[s0:s0 + W + LMAX]
        best = (-1, 0)
        for L in range(0, LMAX + 1):
            yy = ys[L:L + W]
            if xs.std() < 1e-12 or yy.std() < 1e-12:
                continue
            r = np.corrcoef(xs, yy)[0, 1]
            if r > best[0]:
                best = (r, L)
        n = 256
        Xf = np.fft.rfft(xs - xs.mean(), n); Yf = np.fft.rfft(ys[:W] - ys[:W].mean(), n)
        G = Yf * np.conj(Xf); G /= np.abs(G) + 1e-12
        cc = np.fft.irfft(G, n)[:LMAX + 1]
        win_rows.append(dict(heat=k, group=GROUP[k], t_center=s0 + W / 2, Q_mean=d["Q"][s0:s0 + W].mean(),
                             lag_xcorr=best[1], r_xcorr=best[0], lag_gccphat=int(np.argmax(cc))))
wdf = pd.DataFrame(win_rows)
wdf.to_csv(os.path.join(TAB, "q2_delay_windows.csv"), index=False)
good = wdf[wdf.r_xcorr > 0.3]
ts = theilslopes(good.lag_xcorr, 1 / good.Q_mean) if len(good) > 5 else (np.nan, np.nan, np.nan, np.nan)
sp_w = spearmanr(good.Q_mean, good.lag_xcorr) if len(good) > 5 else None

# C: heat-level best constant delay vs mean flow
heat_rows = []
for i, k in enumerate(HEATS):
    d = D[k]; m = eval_mask(d)
    Lk = L_GRID[np.argmax(R2_c[:, i])]
    onset_gas = d["t"][np.argmax(d["co2"] > 1.0)]
    heat_rows.append(dict(heat=k, group=GROUP[k], Q_mean_active=d["Q"][m].mean(), best_const_delay_s=Lk,
                          transport_tau_mean_s=np.nanmean(tau_of(d, TAU0_USE, K_USE)[m]),
                          gas_onset_after_blow_start_s=onset_gas))
hdf = pd.DataFrame(heat_rows)
hdf.to_csv(os.path.join(TAB, "q2_delay_by_heat.csv"), index=False)
sp_h = spearmanr(hdf.Q_mean_active, hdf.best_const_delay_s)
aux = pd.DataFrame([
    dict(method="B: windowed xcorr, Theil-Sen tau = a + b/Q", n=len(good), estimate_a=ts[1], estimate_b=ts[0],
         spearman_Q_vs_tau=sp_w.statistic if sp_w else np.nan, p_value=sp_w.pvalue if sp_w else np.nan),
    dict(method="B: windowed xcorr, median lag (s)", n=len(good), estimate_a=good.lag_xcorr.median(), estimate_b=np.nan,
         spearman_Q_vs_tau=np.nan, p_value=np.nan),
    dict(method="C: heat-level best constant delay vs mean flow", n=len(hdf), estimate_a=hdf.best_const_delay_s.median(),
         estimate_b=np.nan, spearman_Q_vs_tau=sp_h.statistic, p_value=sp_h.pvalue),
    dict(method="D: gas onset after blow start (upper bound, s)", n=len(hdf),
         estimate_a=hdf.gas_onset_after_blow_start_s.median(), estimate_b=np.nan, spearman_Q_vs_tau=np.nan, p_value=np.nan),
])
# E: lance-flow step events. Flame (log total intensity) responds almost immediately to a step, CO responds
# after the transport delay; delay = t_CO - t_flame, kept only when the flame responds within 15 s.
from scipy.ndimage import uniform_filter1d
ev_rows = []
for k in HEATS:
    if k == 5:
        continue                                    # noisy flow record (1 s sampling), excluded
    d = D[k]; Qg = d["Q"]; w = 12
    dd = np.array([Qg[i:i + w].mean() - Qg[i - w:i].mean() if w <= i < len(Qg) - w else 0 for i in range(len(Qg))])
    i = 0
    while i < len(dd):
        if abs(dd[i]) > 0.8:
            j = i + int(np.argmax(np.abs(dd[i:i + 20]))); sgn = np.sign(dd[j])
            def resp(y, lo=-20, hi=150):
                a_, b_ = max(0, j + lo), min(len(y) - 1, j + hi)
                dy = np.gradient(uniform_filter1d(y[a_:b_], 7)) * sgn
                return a_ + int(np.argmax(dy)) - j
            tf, tc = resp(uniform_filter1d(d["logT"], 5)), resp(d["co"])
            ev_rows.append(dict(heat=k, t_step_s=j, Q_before=Qg[j - w:j].mean(), Q_after=Qg[j:j + w].mean(),
                                flame_response_s=tf, co_response_s=tc, delay_s=tc - tf,
                                valid=(abs(tf) <= 15) and (tc - tf > 0) and (j >= 120)))
            i = j + 40
        else:
            i += 1
edf = pd.DataFrame(ev_rows)
edf.to_csv(os.path.join(TAB, "q2_delay_events.csv"), index=False)
ev_ok = edf[edf.valid]
aux = pd.concat([aux, pd.DataFrame([dict(method="E: flow-step events after 120 s (flame responds within 15 s), median delay (s)",
                                         n=len(ev_ok), estimate_a=ev_ok.delay_s.median(), estimate_b=np.nan,
                                         spearman_Q_vs_tau=spearmanr(ev_ok.Q_after, ev_ok.delay_s).statistic if len(ev_ok) > 3 else np.nan,
                                         p_value=spearmanr(ev_ok.Q_after, ev_ok.delay_s).pvalue if len(ev_ok) > 3 else np.nan)])])
aux.to_csv(os.path.join(TAB, "q2_delay_auxiliary.csv"), index=False)
print(edf.round(2))
print(aux)

# ---------------------------------------------------------------- sensitivity (1-D searches)
sens = []
def best_K_and_L(S_, D_backup=None):
    cK = np.array([np.nanmean(loho_r2(S_, targets_for("co", TAU0_USE, K))) for K in K_GRID])
    cL = np.array([np.nanmean(loho_r2(S_, targets_for("co", L=L))) for L in L_GRID[::2]])
    return K_GRID[np.nanargmax(cK)], L_GRID[::2][np.nanargmax(cL)], cK.max(), cL.max()

for npc in (5, 20):
    Kb, Lb, cK, cL = best_K_and_L(ridge_solvers(fold_features(npc)))
    sens.append(dict(setting=f"{npc} spectral PCs", K_best=Kb, L_best=Lb, cvR2_transport=cK, cvR2_const=cL))
for al in (1.0, 100.0):
    Kb, Lb, cK, cL = best_K_and_L(ridge_solvers(F, al))
    sens.append(dict(setting=f"ridge alpha = {al:g}", K_best=Kb, L_best=Lb, cvR2_transport=cK, cvR2_const=cL))
for ts0 in (30.0, 120.0):
    T_START = ts0
    Kb, Lb, cK, cL = best_K_and_L(ridge_solvers(fold_features(10)))
    sens.append(dict(setting=f"evaluation starts at {ts0:g} s", K_best=Kb, L_best=Lb, cvR2_transport=cK, cvR2_const=cL))
T_START = 60.0
D[5] = load_heat(5, dt_override=0.5)
Kb, Lb, cK, cL = best_K_and_L(ridge_solvers(fold_features(10)))
sens.append(dict(setting="heat 5 with dt = 0.5 s", K_best=Kb, L_best=Lb, cvR2_transport=cK, cvR2_const=cL))
D[5] = load_heat(5)
sens.insert(0, dict(setting="baseline (10 PCs, alpha 10, start 60 s)", K_best=K_hat, L_best=L_hat,
                    cvR2_transport=crit2[i0].max(), cvR2_const=critc.max()))
pd.DataFrame(sens).to_csv(os.path.join(TAB, "q2_delay_sensitivity.csv"), index=False)
print(pd.DataFrame(sens))

# save per-heat tau(t) at native sampling for Q3
for k in HEATS:
    pr = np.load(os.path.join(CLEAN, f"heat{k:02d}_proc.npz"))
    d = D[k]
    tau = tau_of(d, TAU0_USE, K_USE)
    np.savez_compressed(os.path.join(CLEAN, f"heat{k:02d}_delay.npz"), t=d["t"], tau=tau,
                        tau_native=np.interp(pr["t"], d["t"], np.where(np.isnan(tau), np.nanmax(tau), tau)),
                        tau0=TAU0_USE, K=K_USE, L_const=L_hat)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(7, 4.6))
qq = np.linspace(2.5, 7.5, 200)
tqq = boot1[:, None] / qq[None, :]
ax.fill_between(qq, np.percentile(tqq, 2.5, 0), np.percentile(tqq, 97.5, 0), color="C0", alpha=0.2,
                label="95% bootstrap band")
ax.plot(qq, K_USE / qq, "C0", lw=2, label=fr"Transport model $\tau = {K_USE:.0f}/Q$")
for g in ("6B", "6C"):
    gg = good[good.group == g]
    if len(good) >= 5:
        ax.scatter(gg.Q_mean, gg.lag_xcorr, s=12, alpha=0.4, color=GROUP_COLOR[g], label=f"Window estimates ({g})")
# binned medians of the window estimates (interquartile whiskers)
bins = [0, 4.5, 5.0, 5.5, 10]
gb = good.groupby(pd.cut(good.Q_mean, bins), observed=True)
bin_tab = pd.DataFrame(dict(Q_bin=[str(i) for i in gb.groups.keys()], Q_center=gb.Q_mean.median().values,
                            delay_median=gb.lag_xcorr.median().values, q25=gb.lag_xcorr.quantile(0.25).values,
                            q75=gb.lag_xcorr.quantile(0.75).values, n=gb.size().values))
bin_tab.to_csv(os.path.join(TAB, "q2_delay_window_bins.csv"), index=False)
ax.errorbar(bin_tab.Q_center, bin_tab.delay_median, yerr=[bin_tab.delay_median - bin_tab.q25, bin_tab.q75 - bin_tab.delay_median],
            fmt="s-", color="k", ms=7, capsize=4, lw=1.5, label="Window estimates: binned median (IQR)")
ax.scatter(ev_ok.Q_after, ev_ok.delay_s, s=70, marker="*", color="k", label="Flow-step events")
ax.set_xlabel("Lance oxygen flow Q (m$^3$/s)"); ax.set_ylabel("Flue-gas delay (s)")
ax.set_xlim(2.5, 7.5); ax.set_ylim(-2, 100); ax.legend(fontsize=7, ncol=1, loc="upper right")
ax.set_title("Flue-gas delay versus lance flow")
savefig(fig, "q2_delay_vs_flow")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
im = axs[0].imshow(crit2, origin="lower", aspect="auto", cmap="viridis",
                   extent=[K_GRID[0], K_GRID[-1], T0_GRID[0], T0_GRID[-1]])
axs[0].plot(K_hat, tau0_hat, "r*", ms=14, label="Optimum")
axs[0].set_xlabel("K (m$^3$)"); axs[0].set_ylabel(r"$\tau_0$ (s)"); axs[0].legend(); axs[0].grid(False)
fig.colorbar(im, ax=axs[0], label="Mean LOHO $R^2$")
axs[0].set_title("Transport model: cross-validated criterion (CO marker)")
axs[1].plot(L_GRID, critc, "C1", label="Constant delay")
axs[1].axhline(crit2.max(), color="C0", ls="--", label="Transport model optimum")
axs[1].set_xlabel("Constant delay L (s)"); axs[1].set_ylabel("Mean LOHO $R^2$"); axs[1].legend()
axs[1].set_title("Constant-delay model: cross-validated criterion")
savefig(fig, "q2_delay_criterion")

k = 3; d = D[k]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(d["t"] / 60, d["Q"], "C0"); ax.set_ylabel("Lance flow Q (m$^3$/s)", color="C0")
ax2 = ax.twinx(); ax2.plot(d["t"] / 60, tau_of(d, TAU0_USE, K_USE), "C3"); ax2.set_ylabel("Delay tau (s)", color="C3")
ax2.grid(False); ax.set_xlabel("Time (min)"); ax.set_title(f"Heat {k}: lance flow and time-varying delay")
savefig(fig, "q2_delay_timeseries")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, k in zip(axs, (3, 8)):
    d = D[k]
    z = pca_all.transform(d["Xs"])[:, 0]
    z = (z - z.mean()) / z.std() * np.sign(np.corrcoef(z, d["co"])[0, 1])
    co_a = shifted(d, d["co"], tau_of(d, TAU0_USE, K_USE))
    nz = lambda y: (y - np.nanmean(y)) / np.nanstd(y)
    ax.plot(d["t"] / 60, z, color="0.3", lw=1, label="Flame spectrum PC1 (real time)")
    ax.plot(d["t"] / 60, nz(d["co"]), "C3", lw=1, alpha=0.6, label="CO as measured")
    ax.plot(d["t"] / 60, nz(co_a), "C0", lw=1.3, label="CO shifted by tau(t)")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Standardized value"); ax.set_title(f"Heat {k}")
axs[0].legend(fontsize=7)
fig.suptitle("Effect of delay alignment")
savefig(fig, "q2_alignment_effect")
print("Q2B finished")
