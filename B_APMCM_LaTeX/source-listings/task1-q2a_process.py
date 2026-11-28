"""Problem 2 (part A): O2 / flue-gas outlier removal and continuous reconstruction, blowing stages,
oxygen-gas relations, carbon-oxygen consistency diagnosis.

Outputs: data_clean/heatXX_proc.npz, tables/q2_*.csv, figures/q2_*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, make_smoothing_spline
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.isotonic import IsotonicRegression
from common import (HEATS, HEAT_ID, GROUP, DT, CLEAN, TAB, SEED, GROUP_COLOR,
                    load_raw, hampel_mask, update_points, style, savefig)

style()


def reconstruct_o2(t, o2, win_sec=24.0, dt=0.5):
    """Monotone reconstruction of V(t); lance flow Q(t) = sliding linear-regression slope of V.

    The O2 counter updates about once per second; clock drift against the 0.5 s logger makes it
    skip one update roughly every 11.5 s and then add a double increment (a beat artefact).
    A slope window of ~2 beat periods (24 s) removes this artefact."""
    iso = IsotonicRegression(increasing=True).fit_transform(t, o2)
    n_back = int((np.abs(iso - o2) > 1e-9).sum())
    idx = update_points(iso)
    f = PchipInterpolator(t[idx], iso[idx], extrapolate=False)
    V = f(t)
    V = np.where(np.isnan(V), np.interp(t, t[idx], iso[idx]), V)
    V = np.maximum.accumulate(V)
    w = int(round(win_sec / dt)) | 1
    Q = np.clip(savgol_filter(iso, w, 1, deriv=1, delta=dt, mode="interp"), 0, None)
    return V, Q, n_back, idx


def reconstruct_gas(t, c, Q, dt, hide=None):
    """Return smooth reconstruction and diagnostics. hide: boolean mask of update points held out."""
    c = np.clip(c.astype(float), 0, None)
    bad = np.zeros(len(c), bool)
    # long plateaus inside active blowing
    active = Q > 0.5 * np.median(Q[Q > 0])
    run_start = update_points(c)
    run_end = np.r_[run_start[1:], len(c)]
    n_plateau = 0
    for s, e in zip(run_start, run_end):
        if (e - s) * dt > 10.0 and active[s:e].mean() > 0.5:
            bad[s + 1:e] = True       # keep first sample (a genuine reading), drop the stuck tail
            n_plateau += 1
    # spikes: Hampel filter on the sequence of value updates (sample-and-hold repeats removed, otherwise the
    # local MAD collapses to zero and ordinary changes would be flagged)
    up = np.array([i for i in update_points(c) if not bad[i]])
    x = c[up]
    med = median_filter(x, size=11, mode="nearest")
    mad = median_filter(np.abs(x - med), size=11, mode="nearest")
    noise = 1.4826 * np.median(np.abs(np.diff(x))) / np.sqrt(2)      # global measurement-noise floor
    sp_up = np.abs(x - med) > 4.0 * np.maximum(1.4826 * mad, noise)
    spikes = np.zeros(len(c), bool)
    for i0, i1, f in zip(up, np.r_[up[1:], len(c)], sp_up):
        if f:
            spikes[i0:i1] = True
    bad |= spikes
    idx = np.array([i for i in update_points(c) if not bad[i]])
    use = idx if hide is None else idx[~hide]
    f = PchipInterpolator(t[use], c[use], extrapolate=False)
    y = f(t)
    y = np.where(np.isnan(y), np.interp(t, t[use], c[use]), y)
    sp = make_smoothing_spline(t, y)
    return np.clip(sp(t), 0, None), dict(plateaus=n_plateau, plateau_pts=int(bad.sum() - spikes.sum()),
                                         spikes=int(sp_up.sum()), update_pts=len(idx)), idx


rows, hold_rows, mb_rows = [], [], []
P = {}
for k in HEATS:
    r = load_raw(k)
    t, dt = r["t"], DT[k]
    V, Q, nb, o2idx = reconstruct_o2(t, r["o2"], dt=dt)
    Qraw = np.gradient(r["o2"], t)
    co, dco, idx_co = reconstruct_gas(t, r["co"], Q, dt)
    co2, dco2, idx_co2 = reconstruct_gas(t, r["co2"], Q, dt)
    # hold-out validation (10 % of update points, 5 repetitions)
    for name, raw, idx in (("CO", r["co"], idx_co), ("CO2", r["co2"], idx_co2)):
        errs = []
        for rep in range(5):
            g = np.random.default_rng(SEED + rep)
            h = g.random(len(idx)) < 0.10
            h[0] = h[-1] = False
            yh, _, _ = reconstruct_gas(t, raw, Q, dt, hide=h)
            errs.append(np.sqrt(np.mean((yh[idx[h]] - raw[idx[h]]) ** 2)))
        hold_rows.append(dict(heat=k, signal=name, holdout_rmse=np.mean(errs), sd=np.std(errs),
                              signal_mean=raw[raw > 0].mean()))
    Eco = cumulative_trapezoid(co, t, initial=0)
    Eco2 = cumulative_trapezoid(co2, t, initial=0)
    P[k] = dict(t=t, V=V, Q=Q, Qraw=Qraw, co=co, co2=co2, Eco=Eco, Eco2=Eco2,
                co_raw=r["co"], co2_raw=r["co2"], o2_raw=r["o2"])
    rows.append(dict(heat=k, heat_id=HEAT_ID[k], group=GROUP[k], dt_s=dt, duration_min=t[-1] / 60,
                     o2_backsteps=nb, o2_update_pts=len(o2idx), Q_median=np.median(Q[Q > 0.5]),
                     Q_p10=np.percentile(Q[Q > 0.5], 10), Q_p90=np.percentile(Q[Q > 0.5], 90),
                     rawdiff_Q_std=np.std(Qraw), recon_Q_std=np.std(Q),
                     co_plateaus=dco["plateaus"], co_plateau_pts=dco["plateau_pts"], co_spikes=dco["spikes"],
                     co2_plateaus=dco2["plateaus"], co2_plateau_pts=dco2["plateau_pts"],
                     co2_spikes=dco2["spikes"]))
    O = V[-1]
    mb_rows.append(dict(heat=k, V_end_m3=O, E_CO_m3=Eco[-1], E_CO2_m3=Eco2[-1],
                        O2_needed_m3=0.5 * Eco[-1] + Eco2[-1], ratio_needed_to_supplied=(0.5 * Eco[-1] + Eco2[-1]) / O,
                        CO_plus_CO2_median_active=np.median((co + co2)[Q > 0.5 * np.median(Q)]),
                        CO_plus_CO2_max=(co + co2).max()))

pd.DataFrame(rows).to_csv(os.path.join(TAB, "q2_reconstruction_summary.csv"), index=False)

beat = []
for k in HEATS:
    r = load_raw(k); d = np.diff(r["o2"]); t = r["t"]
    nz = d[d > 0]; med = np.median(nz)
    dbl = np.where(d > 1.6 * med)[0]
    row = dict(heat=k, single_increment_median=med, double_increments=len(dbl),
               double_spacing_s=np.median(np.diff(dbl)) * DT[k] if len(dbl) > 2 else np.nan)
    for w in (12, 24, 36):
        _, Qw, _, _ = reconstruct_o2(t, r["o2"], win_sec=w, dt=DT[k])
        a = Qw > 0.5 * np.median(Qw)
        row[f"Q_detrended_std_win{w}s"] = np.std(Qw[a] - savgol_filter(Qw, int(60 / DT[k]) | 1, 1)[a])
    beat.append(row)
pd.DataFrame(beat).to_csv(os.path.join(TAB, "q2_o2_counter_beat.csv"), index=False)
print(pd.DataFrame(beat).round(3))
pd.DataFrame(hold_rows).to_csv(os.path.join(TAB, "q2_gas_holdout_error.csv"), index=False)
mb = pd.DataFrame(mb_rows)
mb.to_csv(os.path.join(TAB, "q2_mass_balance.csv"), index=False)
print(mb.round(2))


# ---------------------------------------------------------------- stages: 3-breakpoint piecewise linear CO(V)
def pw_fit(x, y, bps):
    A = np.column_stack([np.ones_like(x), x] + [np.clip(x - b, 0, None) for b in bps])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.sum((A @ coef - y) ** 2), coef


def pw_eval(v, coef, bps):
    return coef[0] + coef[1] * v + sum(c * np.clip(v - b, 0, None) for c, b in zip(coef[2:], bps))


STAGE_NAMES = ["Early blow (CO2-dominated)", "Decarburization ramp-up", "Steady decarburization",
               "End-of-blow decline"]
stage_rows = []
for k in HEATS:
    p = P[k]
    sel = np.arange(0, len(p["t"]), int(2 / DT[k]))
    x, y = p["V"][sel], p["co"][sel]
    Ve = p["V"][-1]
    # physical constraints: ramp-up starts in 5-40 % of progress, ends before 60 %, decline after 75 %
    g1 = np.linspace(0.05 * Ve, 0.40 * Ve, 36)
    g2 = np.linspace(0.15 * Ve, 0.60 * Ve, 46)
    g3 = np.linspace(0.75 * Ve, 0.99 * Ve, 25)
    best = (np.inf, None, None)
    for b1 in g1:
        for b2 in g2[g2 > b1 + 0.05 * Ve]:
            for b3 in g3[g3 > b2 + 0.03 * Ve]:
                sse, coef = pw_fit(x, y, (b1, b2, b3))
                if sse < best[0]:
                    best = (sse, (b1, b2, b3), coef)
    sse, bps, coef = best
    r2 = 1 - sse / np.sum((y - y.mean()) ** 2)
    ts = [np.interp(b, p["V"], p["t"]) for b in bps]
    p.update(bps=np.array(bps), tbs=np.array(ts), pw_coef=coef)
    slopes = np.cumsum(coef[1:])
    frac_co = (p["co"] / np.maximum(p["co"] + p["co2"], 1e-6))
    stg = np.searchsorted(np.array(bps), p["V"])
    row = dict(heat=k, group=GROUP[k], r2_piecewise=r2)
    for i in range(3):
        row[f"V_break{i + 1}"] = bps[i]; row[f"t_break{i + 1}_s"] = ts[i]; row[f"Vfrac_break{i + 1}"] = bps[i] / Ve
    for i in range(4):
        row[f"slope_stage{i + 1}"] = slopes[i]
        m = stg == i
        row[f"CO_fraction_stage{i + 1}"] = np.mean(frac_co[m & (p["co"] + p["co2"] > 1)]) if m.any() else np.nan
    stage_rows.append(row)
st = pd.DataFrame(stage_rows)
st.to_csv(os.path.join(TAB, "q2_stages.csv"), index=False)
mw = []
for col in ["Vfrac_break1", "Vfrac_break2", "Vfrac_break3"] + [f"slope_stage{i}" for i in range(1, 5)]:
    a, b = st[st.group == "6B"][col], st[st.group == "6C"][col]
    u = mannwhitneyu(a, b)
    mw.append(dict(variable=col, mean_6B=a.mean(), mean_6C=b.mean(), U=u.statistic, p_value=u.pvalue))
pd.DataFrame(mw).to_csv(os.path.join(TAB, "q2_stage_group_test.csv"), index=False)
print(st[["heat", "Vfrac_break1", "Vfrac_break2", "Vfrac_break3", "r2_piecewise",
          "CO_fraction_stage1", "CO_fraction_stage3"]].round(3))


# ---------------------------------------------------------------- relations
def stage_of(p):
    return np.searchsorted(p["bps"], p["V"])


rel_rows = []
for k in HEATS:
    p = P[k]
    s = stage_of(p)
    for gname in ("co", "co2"):
        for si in range(4):
            m = s == si
            x, y = p["V"][m], p[gname][m]
            if m.sum() < 10:
                continue
            for deg in (1, 2):
                c = np.polyfit(x, y, deg)
                r2 = 1 - np.sum((np.polyval(c, x) - y) ** 2) / np.sum((y - y.mean()) ** 2)
                rel_rows.append(dict(heat=k, signal=gname.upper(), stage=si + 1, degree=deg,
                                     slope_at_mean=np.polyval(np.polyder(c), x.mean()), r2=r2, n=m.sum()))
pd.DataFrame(rel_rows).to_csv(os.path.join(TAB, "q2_relation_stagewise.csv"), index=False)


def logistic(v, a, b, c, d):
    return d + a / (1 + np.exp(-(v - b) / c))


cum_rows = []
for k in HEATS:
    p = P[k]
    sel = np.arange(0, len(p["t"]), int(2 / DT[k]))
    v = p["V"][sel]
    for nm in ("Eco", "Eco2"):
        y = p[nm][sel]
        n = len(y)
        try:
            pr, _ = curve_fit(logistic, v, y, p0=[y.max() * 1.2, v.mean(), v.std(), 0], maxfev=20000)
            sse_l = np.sum((logistic(v, *pr) - y) ** 2)
        except RuntimeError:
            sse_l = np.nan
        c = np.polyfit(v, y, 2); sse_q = np.sum((np.polyval(c, v) - y) ** 2)
        sse_p, _ = pw_fit(v, y, p["bps"])
        aic = lambda sse, kp: n * np.log(sse / n) + 2 * kp
        cum_rows.append(dict(heat=k, signal=nm.replace("Eco2", "E_CO2").replace("Eco", "E_CO"),
                             AIC_logistic=aic(sse_l, 4), AIC_quadratic=aic(sse_q, 3), AIC_piecewise=aic(sse_p, 8),
                             R2_logistic=1 - sse_l / np.sum((y - y.mean()) ** 2),
                             R2_piecewise=1 - sse_p / np.sum((y - y.mean()) ** 2)))
pd.DataFrame(cum_rows).to_csv(os.path.join(TAB, "q2_cumulative_fit.csv"), index=False)

# spearman matrix (pooled, sampled every 2 s, active blowing)
names = ["V", "Q", "co", "co2", "Eco", "Eco2"]
labels = ["Cum. O2", "Lance flow", "CO", "CO2", "Total CO", "Total CO2"]
Ms = []
for k in HEATS:
    p = P[k]
    sel = np.arange(0, len(p["t"]), int(2 / DT[k]))
    Ms.append(spearmanr(np.column_stack([p[n][sel] for n in names])).statistic)
Mmean = np.mean(Ms, axis=0)
pd.DataFrame(Mmean, index=labels, columns=labels).to_csv(os.path.join(TAB, "q2_spearman_mean.csv"))

# save processed data
for k in HEATS:
    p = P[k]
    np.savez_compressed(os.path.join(CLEAN, f"heat{k:02d}_proc.npz"),
                        **{n: p[n] for n in ("t", "V", "Q", "co", "co2", "Eco", "Eco2", "co_raw", "co2_raw")},
                        bps=p["bps"], tbs=p["tbs"])

# ---------------------------------------------------------------- figures
for k in HEATS:
    p = P[k]
    tm = p["t"] / 60
    fig, axs = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    ax = axs[0, 0]
    ax.plot(tm, p["Qraw"], color="0.75", lw=0.5, label="Raw difference")
    ax.plot(tm, p["Q"], color="C0", lw=1.5, label="Reconstructed")
    ax.set_ylim(0, max(12, np.percentile(p["Q"], 99) * 1.5)); ax.set_ylabel("Lance O2 flow (m$^3$/s)")
    ax.legend(); ax.set_title("Oxygen-lance flow rate")
    ax = axs[0, 1]
    ax.plot(tm, p["o2_raw"], ".", ms=1.5, color="0.6", label="Raw")
    ax.plot(tm, p["V"], color="C0", lw=1.5, label="Reconstructed")
    ax.set_ylabel("Cumulative O2 (m$^3$)"); ax.legend(); ax.set_title("Cumulative oxygen consumption")
    for ax, raw, rec, ttl, col in ((axs[1, 0], p["co2_raw"], p["co2"], "CO2", "C2"),
                                   (axs[1, 1], p["co_raw"], p["co"], "CO", "C3")):
        ax.plot(tm, raw, ".", ms=1.5, color="0.6", label="Raw")
        ax.plot(tm, rec, color=col, lw=1.5, label="Reconstructed")
        ax.set_ylabel(f"{ttl} (as given, m$^3$/s)"); ax.legend(); ax.set_title(f"{ttl} signal")
    axs[2, 0].plot(tm, p["Eco2"], color="C2", lw=1.5); axs[2, 0].set_ylabel("Total CO2 (m$^3$)")
    axs[2, 0].set_title("Total CO2 emission")
    axs[2, 1].plot(tm, p["Eco"], color="C3", lw=1.5); axs[2, 1].set_ylabel("Total CO (m$^3$)")
    axs[2, 1].set_title("Total CO emission")
    for ax in axs.flat:
        for tb in p["tbs"]:
            ax.axvline(tb / 60, color="0.4", ls=":", lw=0.8)
    for ax in axs[2]:
        ax.set_xlabel("Time (min)")
    fig.suptitle(f"Heat {k} ({HEAT_ID[k]}): reconstructed process signals (dotted: stage breakpoints)")
    fig.tight_layout()
    savefig(fig, f"q2_curves_heat{k:02d}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for k in HEATS:
    p = P[k]
    x = p["V"] / p["V"][-1]
    axs[0].plot(x, p["co"], color=GROUP_COLOR[GROUP[k]], lw=1, alpha=0.8)
    axs[1].plot(x, p["co2"], color=GROUP_COLOR[GROUP[k]], lw=1, alpha=0.8)
for ax, nm in zip(axs, ("CO", "CO2")):
    ax.set_xlabel("Blowing progress V(t)/V_end"); ax.set_ylabel(nm)
    ax.plot([], [], color=GROUP_COLOR["6B"], label="6B heats (1-4)")
    ax.plot([], [], color=GROUP_COLOR["6C"], label="6C heats (5-10)"); ax.legend()
axs[0].set_title("CO versus blowing progress"); axs[1].set_title("CO2 versus blowing progress")
savefig(fig, "q2_curves_overlay")

fig, axs = plt.subplots(2, 5, figsize=(20, 7), sharey=True)
for ax, k in zip(axs.flat, HEATS):
    p = P[k]
    ax.plot(p["V"], p["co"], color="C3", lw=1, label="CO")
    ax.plot(p["V"], p["co2"], color="C2", lw=1, label="CO2")
    vv = np.linspace(p["V"].min(), p["V"].max(), 200)
    ax.plot(vv, pw_eval(vv, p["pw_coef"], p["bps"]), "k--", lw=1, label="Piecewise fit (CO)")
    for b in p["bps"]:
        ax.axvline(b, color="0.5", ls=":")
    ax.set_title(f"Heat {k}"); ax.set_xlabel("Cumulative O2 (m$^3$)")
axs[0, 0].set_ylabel("Gas signal"); axs[1, 0].set_ylabel("Gas signal"); axs[0, 0].legend()
fig.tight_layout()
savefig(fig, "q2_relation_instant")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for k in HEATS:
    p = P[k]
    axs[0].plot(p["V"], p["Eco"], color=GROUP_COLOR[GROUP[k]], lw=1)
    axs[1].plot(p["V"], p["Eco2"], color=GROUP_COLOR[GROUP[k]], lw=1)
axs[0].set_ylabel("Total CO (m$^3$)"); axs[1].set_ylabel("Total CO2 (m$^3$)")
for ax in axs:
    ax.set_xlabel("Cumulative O2 (m$^3$)")
axs[0].set_title("Total CO versus cumulative O2"); axs[1].set_title("Total CO2 versus cumulative O2")
savefig(fig, "q2_relation_cumulative")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(Mmean, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=40, ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels(labels)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{Mmean[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.grid(False); fig.colorbar(im); ax.set_title("Mean Spearman correlation (10 heats)")
savefig(fig, "q2_corr_heatmap")

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(mb))
ax.bar(x, mb["ratio_needed_to_supplied"], color=[GROUP_COLOR[GROUP[k]] for k in HEATS])
ax.axhline(1, color="k", ls="--", lw=1, label="Physical upper bound (ratio = 1)")
ax.set_xticks(x); ax.set_xticklabels([f"H{k}" for k in HEATS])
ax.set_ylabel("O2 required / O2 supplied"); ax.legend()
ax.set_title("Carbon-oxygen balance check under the m$^3$/s interpretation")
savefig(fig, "q2_mass_balance")
print("Q2A finished")
