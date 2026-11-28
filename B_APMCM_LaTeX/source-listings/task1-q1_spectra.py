"""Problem 1: spectral outlier removal (baseline Hampel + RPCA) and continuous reconstruction.

Outputs: data_clean/heatXX_clean.npz, tables/q1_*.csv, figures/q1_*.png
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline, CubicSpline
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter, find_peaks
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu
from common import (HEATS, HEAT_ID, GROUP, DT, SAT, WL_GRID, CLEAN, TAB, SEED,
                    load_raw, hampel_mask, style, savefig)

style()
rng = np.random.default_rng(SEED)
NW = len(WL_GRID)


# ---------------------------------------------------------------- helpers
def to_grid(raw):
    """Interpolate each frame to the unified wavelength grid; return log-intensity and saturation mask."""
    wl, S = raw["wl"], raw["spec"]
    X = np.empty((S.shape[0], NW))
    satf = np.empty((S.shape[0], NW))
    sat = (S >= SAT).astype(float)
    for i in range(S.shape[0]):
        X[i] = np.interp(WL_GRID, wl, np.log(np.clip(S[i], 1, None)))
        satf[i] = np.interp(WL_GRID, wl, sat[i])
    return X, satf > 0


def baseline_detect(X, miss, nsig_frame=5.0, nsig_pix=4.0):
    """P0 detector: frame-level Hampel on log total intensity + pixel spikes in both directions."""
    I = np.exp(np.where(miss, np.nan, X))
    T = np.log(np.nansum(I, axis=1))
    bad_frame = hampel_mask(T, 15, nsig_frame)
    Xf = np.where(miss, np.nan, X)
    # fill missing temporarily along time for filtering
    Xt = pd.DataFrame(Xf).interpolate(limit_direction="both").values
    med_t = median_filter(Xt, size=(11, 1), mode="nearest")
    r_t = Xt - med_t
    s_t = 1.4826 * np.median(np.abs(r_t), axis=0, keepdims=True) + 1e-9
    med_w = median_filter(Xt, size=(1, 5), mode="nearest")
    r_w = Xt - med_w
    s_w = 1.4826 * np.median(np.abs(r_w), axis=1, keepdims=True) + 1e-9
    pix = (np.abs(r_t) > nsig_pix * s_t) & (np.abs(r_w) > nsig_pix * s_w)
    out = pix.copy()
    out[bad_frame, :] = True
    out &= ~miss
    return out, bad_frame


def rpca(X, obs, lam=None, tol=1e-6, max_iter=300):
    """Principal component pursuit by inexact ALM with missing entries (obs=False are free)."""
    m, n = X.shape
    lam = lam or 1.0 / np.sqrt(max(m, n))
    D = np.where(obs, X, 0.0)
    normD = np.linalg.norm(D)
    Y = np.zeros_like(D)
    S = np.zeros_like(D)
    L = np.zeros_like(D)
    mu = 1.25 / np.linalg.norm(D, 2)
    mu_max, rho = mu * 1e7, 1.5
    for it in range(max_iter):
        U, s, Vt = np.linalg.svd(D - S + Y / mu, full_matrices=False)
        s = np.maximum(s - 1.0 / mu, 0)
        r = int((s > 0).sum())
        L = (U[:, :r] * s[:r]) @ Vt[:r]
        T = D - L + Y / mu
        S = np.sign(T) * np.maximum(np.abs(T) - lam / mu, 0)
        S[~obs] = T[~obs]          # unobserved entries absorbed by S
        Z = D - L - S
        Y += mu * Z
        mu = min(mu * rho, mu_max)
        if np.linalg.norm(Z) / normD < tol:
            break
    return L, S, it + 1, r


def robust_lowrank(X, obs, r, k=4.0, n_iter=25):
    """Rank-constrained robust PCA (GoDec-type alternating projection):
    L = best rank-r approximation of (X - S) (missing entries imputed by L),
    S = hard-thresholded residual |X - L| > k * robust sigma (only on observed entries)."""
    Xf = pd.DataFrame(np.where(obs, X, np.nan)).interpolate(limit_direction="both").values
    S = np.zeros_like(Xf)
    for _ in range(n_iter):
        M = Xf - S
        mu = M.mean(0)
        U, s, Vt = randomized_svd(M - mu, n_components=r, random_state=SEED)
        L = mu + (U * s) @ Vt
        R = Xf - L
        sig = 1.4826 * np.median(np.abs(R[obs] - np.median(R[obs])))
        S_new = np.where(obs & (np.abs(R) > k * sig), R, 0.0)
        Xf = np.where(obs, X, L)
        if np.array_equal(S_new != 0, S != 0):
            S = S_new
            break
        S = S_new
    return L, S


def rpca_detect(X, miss, L, k=5.0, frame_z=5.0):
    """Pixel outliers: |X - L| > k * sigma_j with a robust per-wavelength residual scale.
    Shape-anomaly frames: robust z-score of the frame's mean standardized |residual| (clipped at 10) > frame_z.
    (Whole-frame brightness changes are absorbed by L: flame flicker is physical, not an outlier.)"""
    R = np.where(miss, np.nan, X - L)
    sig = 1.4826 * np.nanmedian(np.abs(R - np.nanmedian(R, 0)), 0) + 1e-9
    Z = np.abs(R) / sig
    out = np.nan_to_num(Z) > k
    e = np.nanmean(np.minimum(Z, 10.0), 1)
    med, mad = np.median(e), 1.4826 * np.median(np.abs(e - np.median(e))) + 1e-9
    bad_frame = (e - med) / mad > frame_z
    # genuine glitches are isolated; a run of > 10 frames (5 s) is a real process change (e.g. end of blow)
    runs = np.split(np.arange(len(bad_frame)), np.where(np.diff(bad_frame.astype(int)) != 0)[0] + 1)
    for rr in runs:
        if bad_frame[rr[0]] and len(rr) > 10:
            bad_frame[rr] = False
            out[rr, :] = np.nan_to_num(Z[rr, :]) > 2 * k     # keep only extreme spikes there
    # inside an anomalous frame only the deviating pixels (|z| > 2.5) are removed; the remaining pixels still
    # determine the frame's low-rank scores (dropping whole frames forces temporal interpolation across flicker)
    out[bad_frame, :] = np.nan_to_num(Z[bad_frame, :]) > 2.5
    out &= ~miss
    return out, bad_frame


def lowrank_complete(X, obs, r, n_iter=40):
    """Rank-r matrix completion by iterative hard-impute SVD (column means kept separate)."""
    Xf = pd.DataFrame(np.where(obs, X, np.nan)).interpolate(limit_direction="both").values
    for _ in range(n_iter):
        mu = Xf.mean(0)
        U, s, Vt = randomized_svd(Xf - mu, n_components=r, random_state=SEED)
        Lr = mu + (U * s) @ Vt
        Xf = np.where(obs, X, Lr)
    return Lr, mu, U * s, Vt


def _whit(n, lam):
    D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    return splu((identity(n) + lam * (D.T @ D)).tocsc())


def separable_spline(X, obs, lam_t=10.0, lam_w=1.0):
    """2-D Whittaker smoother (discrete cubic smoothing spline) applied along time, then wavelength."""
    Xf = pd.DataFrame(np.where(obs, X, np.nan)).interpolate(limit_direction="both").values
    A = _whit(X.shape[0], lam_t).solve(Xf)
    return _whit(X.shape[1], lam_w).solve(A.T).T


def lin_time(X, obs):
    return pd.DataFrame(np.where(obs, X, np.nan)).interpolate(limit_direction="both").values


def sg_time(X, obs):
    return savgol_filter(lin_time(X, obs), 11, 2, axis=0)


def holdout_eval(X, valid, method, frac=0.10, reps=3, **kw):
    """Pixel hold-out: hide random valid pixels (tests the wavelength/time completion)."""
    out = []
    for rep in range(reps):
        r = np.random.default_rng(SEED + rep)
        hide = valid & (r.random(X.shape) < frac)
        P = method(X, valid & ~hide, **kw)
        e = P[hide] - X[hide]
        rel = np.mean(np.abs(np.exp(P[hide]) - np.exp(X[hide])) / np.exp(X[hide]))
        out.append((np.sqrt(np.mean(e ** 2)), rel))
    return np.mean(out, axis=0), np.std(out, axis=0)


def frame_holdout(X, valid, t, method, frac=0.10, reps=3, **kw):
    """Frame hold-out: hide whole frames (tests continuity of the reconstruction in time)."""
    out = []
    for rep in range(reps):
        r = np.random.default_rng(SEED + 10 + rep)
        hf = r.random(X.shape[0]) < frac
        hf[:5] = hf[-5:] = False
        P = method(X, valid, t, hf, **kw)
        m = valid & hf[:, None]
        e = P[m] - X[m]
        rel = np.mean(np.abs(np.exp(P[m]) - np.exp(X[m])) / np.exp(X[m]))
        out.append((np.sqrt(np.mean(e ** 2)), rel))
    return np.mean(out, axis=0), np.std(out, axis=0)


def lr_method(X, obs, r=5):
    return lowrank_complete(X, obs, r)[0]


def fr_linear(X, valid, t, hf):
    return lin_time(X, valid & ~hf[:, None])


def fr_whittaker(X, valid, t, hf, lam=10.0):
    Xf = lin_time(X, valid & ~hf[:, None])
    return _whit(X.shape[0], lam).solve(Xf)


def score_spline(tk, A, tq, lam):
    out = []
    for j in range(A.shape[1]):
        if lam == 0:
            out.append(CubicSpline(tk, A[:, j])(tq))
        else:
            out.append(make_smoothing_spline(tk, A[:, j], lam=lam)(tq))
    return np.column_stack(out)


def fr_lowrank(X, valid, t, hf, r=5, lam=None):
    keep = ~hf
    Lr, mu, A, Vt = lowrank_complete(X[keep], valid[keep], r)
    return mu + score_spline(t[keep], A, t, lam) @ Vt


def inject(Xc, sig_j, rng):
    """Add 1 % pixel spikes (5-10 sigma_j) and 1 % shape-distorted frames (Gaussian bump in log-spectrum)."""
    Xi = Xc.copy()
    spk = rng.random(Xi.shape) < 0.01
    amp = rng.uniform(5, 10, Xi.shape) * sig_j[None, :] * rng.choice([-1, 1], Xi.shape)
    Xi[spk] += amp[spk]
    fr = rng.random(Xi.shape[0]) < 0.01
    for i in np.where(fr)[0]:
        c, w, h = rng.uniform(380, 980), rng.uniform(20, 60), rng.uniform(0.2, 0.5) * rng.choice([-1, 1])
        Xi[i] += h * np.exp(-0.5 * ((WL_GRID - c) / w) ** 2)
    return Xi, spk, fr


def prf(pred, truth):
    tp = (pred & truth).sum(); fp = (pred & ~truth).sum(); fn = (~pred & truth).sum()
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


# ---------------------------------------------------------------- main loop
rows_out, rows_rec, rows_fr, rows_sens, comp_store, inj_rows = [], [], [], [], {}, []

# rank selection (pixel hold-out) and temporal smoothing selection (frame hold-out) on heats 3 and 8
rank_rows, lam_rows = [], []
LAMS = [0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 1e5, None]   # 0 = interpolating cubic spline, None = GCV
for k in (3, 8):
    X, miss = to_grid(load_raw(k))
    t = np.arange(X.shape[0]) * DT[k]
    L, S = robust_lowrank(X, ~miss, 8)
    out, _ = rpca_detect(X, miss, L)
    valid = ~miss & ~out
    for r in range(1, 11):
        (rm, rel), _ = holdout_eval(X, valid, lr_method, reps=1, r=r)
        rank_rows.append(dict(heat=k, rank=r, holdout_rmse_log=rm, rel_err=rel))
    for lam in LAMS:
        (rm, rel), _ = frame_holdout(X, valid, t, fr_lowrank, reps=2, r=5, lam=lam)
        lam_rows.append(dict(heat=k, lam=("GCV" if lam is None else lam), frame_rmse_log=rm, rel_err=rel))
rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(os.path.join(TAB, "q1_rank_selection.csv"), index=False)
piv = rank_df.pivot(index="rank", columns="heat", values="holdout_rmse_log").mean(1)
R_CHOICE = int(piv.idxmin())
lam_df = pd.DataFrame(lam_rows)
lam_df.to_csv(os.path.join(TAB, "q1_time_smoothing_selection.csv"), index=False)
lp = lam_df.groupby("lam", sort=False)["frame_rmse_log"].mean()
# model: the frame hold-out optimum (GCV); display: a strongly smoothed trend surface (lambda = 1e4)
LAM_SEL = None if lp.idxmin() == "GCV" else float(lp.idxmin())
LAM_TREND = 1e4
print("rank selection:", piv.round(4).to_dict(), "-> r =", R_CHOICE)
print("time smoothing:", lp.round(4).to_dict(), "-> lam =", LAM_SEL)

X, miss = to_grid(load_raw(3))
L, S = robust_lowrank(X, ~miss, R_CHOICE)
out, _ = rpca_detect(X, miss, L)
valid = ~miss & ~out
hide = valid & (np.random.default_rng(999).random(X.shape) < 0.10)
best_lam = (np.inf, None)
for lt in (0.1, 1, 10, 100):
    for lw in (0.01, 0.1, 1, 10, 100):
        P_ = separable_spline(X, valid & ~hide, lt, lw)
        e = np.sqrt(np.mean((P_[hide] - X[hide]) ** 2))
        if e < best_lam[0]:
            best_lam = (e, (lt, lw))
LAM_T, LAM_W = best_lam[1]
print("Whittaker lambdas:", LAM_T, LAM_W)

for k in HEATS:
    t0 = time.time()
    raw = load_raw(k)
    X, miss = to_grid(raw)
    t = np.arange(X.shape[0]) * DT[k]
    out0, bf0 = baseline_detect(X, miss)
    _, _, it, rr = rpca(X, ~miss)                       # classical PCP, diagnostic only
    L, S = robust_lowrank(X, ~miss, R_CHOICE)
    out1, bf1 = rpca_detect(X, miss, L)
    for kk in (4.0, 5.0, 6.0):
        o, b = rpca_detect(X, miss, L, k=kk)
        rows_sens.append(dict(heat=k, threshold_sigma=kk, outlier_pixels=int(o.sum()),
                              outlier_frac=o.mean(), bad_frames=int(b.sum())))
    rows_out.append(dict(heat=k, heat_id=HEAT_ID[k], group=GROUP[k], frames=X.shape[0],
                         saturated_pixels=int(miss.sum()), saturated_frac=miss.mean(),
                         baseline_bad_frames=int(bf0.sum()), baseline_outlier_pixels=int(out0.sum()),
                         baseline_outlier_frac=out0.mean(), pcp_rank=rr, robust_rank=R_CHOICE,
                         robust_bad_frames=int(bf1.sum()), robust_outlier_pixels=int(out1.sum()),
                         robust_outlier_frac=out1.mean()))
    valid = ~miss & ~out1
    # pixel hold-out
    pm = {"Linear interpolation (time)": (lin_time, {}),
          "Savitzky-Golay (time)": (sg_time, {}),
          "2-D Whittaker smoothing spline": (separable_spline, {"lam_t": LAM_T, "lam_w": LAM_W}),
          f"Robust low-rank completion (r={R_CHOICE})": (lr_method, {"r": R_CHOICE})}
    for name, (fn, kw) in pm.items():
        (rm, rel), (sd, _) = holdout_eval(X, valid, fn, **kw)
        rows_rec.append(dict(heat=k, method=name, rmse_log=rm, rmse_log_sd=sd, rel_err=rel))
    (rm, rel), (sd, _) = holdout_eval(X, valid, lambda Xa, obs: lowrank_complete(Xa, obs | out1, R_CHOICE)[0])
    rows_rec.append(dict(heat=k, method=f"Low-rank completion without outlier removal", rmse_log=rm,
                         rmse_log_sd=sd, rel_err=rel))
    # frame hold-out
    fm = {"Linear interpolation between frames": (fr_linear, {}),
          "Whittaker smoothing (time)": (fr_whittaker, {"lam": LAM_T}),
          "Low-rank + interpolating spline": (fr_lowrank, {"r": R_CHOICE, "lam": 0}),
          "Low-rank + CV-selected smoothing spline": (fr_lowrank, {"r": R_CHOICE, "lam": LAM_SEL})}
    for name, (fn, kw) in fm.items():
        (rm, rel), (sd, _) = frame_holdout(X, valid, t, fn, **kw)
        rows_fr.append(dict(heat=k, method=name, rmse_log=rm, rmse_log_sd=sd, rel_err=rel))

    # final continuous model: per-frame clean matrix (flicker kept) + continuous-in-time scores
    Lr, mu, A, Vt = lowrank_complete(X, valid, R_CHOICE)
    X_lr = np.where(valid, X, Lr)                     # cleaned per-frame spectra (causal, used by Q2/Q3)
    As = score_spline(t, A, t, LAM_SEL)
    Xrec = mu + As @ Vt
    At = score_spline(t, A, t, LAM_TREND)
    Xtrend = mu + At @ Vt
    flicker_sd = float(np.std((X_lr - Xtrend)[valid]))
    sv = np.linalg.svd(X_lr - X_lr.mean(0), compute_uv=False)
    np.savez_compressed(os.path.join(CLEAN, f"heat{k:02d}_spec.npz"), t=t, wl=WL_GRID,
                        X_obs=X.astype(np.float32), valid=valid, miss_sat=miss, outlier=out1,
                        X_lr=X_lr.astype(np.float32), X_rec=Xrec.astype(np.float32),
                        X_trend=Xtrend.astype(np.float32), flicker_sd=flicker_sd, mu=mu,
                        scores_raw=A, scores=As, loadings=Vt, sv=sv,
                        L_rpca=L.astype(np.float32), S_rpca=S.astype(np.float32))
    rows_out[-1]["flicker_sd_log"] = flicker_sd
    comp_store[k] = dict(t=t, X=X, L=L, S=S, Xrec=Xrec, Xtrend=Xtrend, out=out1, miss=miss, mu=mu, As=As, A=A, Vt=Vt, sv=sv)

    if k in (3, 8):
        Rv = np.where(valid, X - Lr, np.nan)
        sig_j = 1.4826 * np.nanmedian(np.abs(Rv - np.nanmedian(Rv, 0)), 0)
        for rep in range(3):
            g = np.random.default_rng(100 + rep)
            Xi, spk, fr = inject(X_lr, sig_j, g)
            o0, b0 = baseline_detect(Xi, miss)
            Li, _ = robust_lowrank(Xi, ~miss, R_CHOICE)
            o1, b1 = rpca_detect(Xi, miss, Li)
            rowsel = ~fr[:, None] & ~miss
            corrupt = (spk | fr[:, None]) & ~miss
            rec_none = lowrank_complete(Xi, ~miss, R_CHOICE)[0]
            for nm, o, bfr in (("Hampel baseline", o0, b0), ("Robust low-rank", o1, b1)):
                p, rc, f1 = prf(o[rowsel], spk[rowsel])
                fp_, fr_, ff_ = prf(bfr, fr)
                rec = lowrank_complete(Xi, ~miss & ~o, R_CHOICE)[0]
                inj_rows.append(dict(heat=k, rep=rep, method=nm, spike_precision=p, spike_recall=rc, spike_f1=f1,
                                     frame_precision=fp_, frame_recall=fr_, frame_f1=ff_,
                                     recon_rmse_corrupted=np.sqrt(np.mean((rec - X_lr)[corrupt] ** 2)),
                                     recon_rmse_all=np.sqrt(np.mean((rec - X_lr)[~miss] ** 2)),
                                     norm_recon_rmse_corrupted=np.sqrt(np.mean((rec_none - X_lr)[corrupt] ** 2)),
                                     norm_recon_rmse_all=np.sqrt(np.mean((rec_none - X_lr)[~miss] ** 2))))
    print(f"heat {k}: {time.time() - t0:.1f}s; PCP rank {rr}; outliers {out1.mean():.4f}; "
          f"bad frames {bf1.sum()}; sat {miss.mean():.4f}", flush=True)

pd.DataFrame(rows_out).to_csv(os.path.join(TAB, "q1_outlier_summary.csv"), index=False)
rec = pd.DataFrame(rows_rec)
rec.to_csv(os.path.join(TAB, "q1_reconstruction_error.csv"), index=False)
rec.groupby("method")[["rmse_log", "rel_err"]].agg(["mean", "std"]).to_csv(
    os.path.join(TAB, "q1_reconstruction_error_summary.csv"))
frd = pd.DataFrame(rows_fr)
frd.to_csv(os.path.join(TAB, "q1_frame_holdout_error.csv"), index=False)
frd.groupby("method")[["rmse_log", "rel_err"]].agg(["mean", "std"]).to_csv(
    os.path.join(TAB, "q1_frame_holdout_summary.csv"))
print(frd.groupby("method")[["rmse_log", "rel_err"]].mean())
pd.DataFrame(rows_sens).to_csv(os.path.join(TAB, "q1_threshold_sensitivity.csv"), index=False)
inj = pd.DataFrame(inj_rows)
inj.to_csv(os.path.join(TAB, "q1_injection_test.csv"), index=False)
print(rec.groupby("method")[["rmse_log", "rel_err"]].mean())
print(inj.groupby("method")[[c for c in inj.columns if c not in ("heat", "rep", "method")]].mean().T)

# ---------------------------------------------------------------- peaks of the mean spectrum
allmean = np.mean([np.exp(comp_store[k]["mu"]) for k in HEATS], axis=0)
lm = np.log(allmean)
base = median_filter(lm, size=41, mode="nearest")
pk, prop = find_peaks(lm - base, prominence=0.05)
peaks = pd.DataFrame(dict(wavelength_nm=WL_GRID[pk], prominence_log=prop["prominences"]))
peaks.sort_values("prominence_log", ascending=False).head(15).to_csv(
    os.path.join(TAB, "q1_emission_peaks.csv"), index=False)

# ---------------------------------------------------------------- figures
def surface(ax, k, ds_t=300, ds_w=300):
    c = comp_store[k]
    t = c["t"] / 60
    ti = np.linspace(0, len(t) - 1, min(ds_t, len(t))).astype(int)
    wi = np.linspace(0, NW - 1, ds_w).astype(int)
    Z = c["Xtrend"][np.ix_(ti, wi)] / np.log(10)
    TT, WW = np.meshgrid(t[ti], WL_GRID[wi], indexing="ij")
    ax.plot_surface(TT, WW, Z, cmap="inferno", linewidth=0, antialiased=True, rcount=200, ccount=200)
    ax.view_init(elev=25, azim=-55)
    ax.set_box_aspect((1.5, 1.2, 0.75))
    ax.set_xlabel("Time (min)", labelpad=6)
    ax.set_ylabel("Wavelength (nm)", labelpad=8)
    ax.set_zlabel("log$_{10}$ Intensity", labelpad=6)


for k in HEATS:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surface(ax, k)
    ax.set_title(f"Heat {k} ({HEAT_ID[k]}): continuous flame spectrum (trend surface)")
    savefig(fig, f"q1_surface_heat{k:02d}")

fig = plt.figure(figsize=(20, 8))
for i, k in enumerate(HEATS):
    ax = fig.add_subplot(2, 5, i + 1, projection="3d")
    surface(ax, k, 200, 200)
    ax.set_title(f"Heat {k} ({HEAT_ID[k]})", fontsize=10)
    ax.tick_params(labelsize=7)
fig.tight_layout()
savefig(fig, "q1_surface_grid")

for k in (1,):
    c = comp_store[k]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow((c["Xtrend"] / np.log(10)).T, aspect="auto", origin="lower", cmap="inferno",
                   extent=[0, c["t"][-1] / 60, WL_GRID[0], WL_GRID[-1]])
    fig.colorbar(im, label="log10 Intensity")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Wavelength (nm)")
    ax.set_title(f"Heat {k}: reconstructed spectrum (heat map)")
    ax.grid(False)
    savefig(fig, f"q1_heatmap_heat{k:02d}")

# outlier example: heat with most outliers
kx = max(HEATS, key=lambda k: comp_store[k]["out"].sum())
c = comp_store[kx]
j = int(np.argmin(np.abs(WL_GRID - 600)))
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
tt = c["t"] / 60
axs[0].plot(tt, c["X"][:, j] / np.log(10), ".", ms=2, color="0.6", label="Observed")
o = c["out"][:, j]
axs[0].plot(tt[o], c["X"][o, j] / np.log(10), "o", ms=4, mfc="none", color="red", label="Detected outlier")
axs[0].plot(tt, c["Xrec"][:, j] / np.log(10), "-", color="C0", lw=1.0, label="Continuous model (GCV)")
axs[0].plot(tt, c["Xtrend"][:, j] / np.log(10), "-", color="C1", lw=2.0, label="Trend surface")
axs[0].set_xlabel("Time (min)"); axs[0].set_ylabel("log10 Intensity")
axs[0].set_title(f"Heat {kx}, wavelength {WL_GRID[j]:.0f} nm"); axs[0].legend()
mid = (c["t"] > 0.2 * c["t"][-1]) & (c["t"] < 0.85 * c["t"][-1]) & (c["out"].mean(1) > 0.9)
dev = np.abs(c["X"] - c["Xrec"]).mean(1)
fi = int(np.argmax(np.where(mid, dev, -1)))
axs[1].plot(WL_GRID, c["X"][fi] / np.log(10), color="0.6", lw=0.8, label="Observed frame")
axs[1].plot(WL_GRID[c["out"][fi]], c["X"][fi, c["out"][fi]] / np.log(10), ".", color="red", ms=3,
            label="Detected outlier")
axs[1].plot(WL_GRID, c["Xrec"][fi] / np.log(10), color="C0", lw=1.5, label="Reconstruction")
axs[1].set_xlabel("Wavelength (nm)"); axs[1].set_ylabel("log10 Intensity")
axs[1].set_title(f"Heat {kx}, frame at t = {c['t'][fi] / 60:.2f} min"); axs[1].legend()
savefig(fig, "q1_outlier_example")

# RPCA decomposition
c = comp_store[3]
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
ext = [0, c["t"][-1] / 60, WL_GRID[0], WL_GRID[-1]]
Xs = np.where(c["miss"], np.nan, c["X"])
for ax, M, ttl, cm in ((axs[0], Xs, "Observed log-intensity X", "inferno"),
                       (axs[1], c["L"], "Low-rank component L", "inferno"),
                       (axs[2], np.where(c["miss"], np.nan, c["S"]), "Sparse component S", "RdBu_r")):
    if cm == "RdBu_r":
        v = np.nanpercentile(np.abs(M), 99.9)
        im = ax.imshow(M.T, aspect="auto", origin="lower", cmap=cm, extent=ext, vmin=-v, vmax=v)
    else:
        im = ax.imshow(M.T, aspect="auto", origin="lower", cmap=cm, extent=ext)
    fig.colorbar(im, ax=ax); ax.set_title(ttl); ax.set_xlabel("Time (min)"); ax.grid(False)
axs[0].set_ylabel("Wavelength (nm)")
fig.suptitle("Heat 3: robust PCA decomposition X = L + S")
savefig(fig, "q1_rpca_decomposition")

# components
fig, axs = plt.subplots(2, 1, figsize=(9, 6.5))
for i in range(min(3, R_CHOICE)):
    axs[0].plot(WL_GRID, c["Vt"][i], label=f"phi_{i + 1}")
    axs[1].plot(c["t"] / 60, c["As"][:, i], label=f"a_{i + 1}(t)")
for p in peaks.sort_values("prominence_log", ascending=False).head(6)["wavelength_nm"]:
    axs[0].axvline(p, color="0.5", ls=":", lw=0.8)
axs[0].set_xlabel("Wavelength (nm)"); axs[0].set_ylabel("Loading"); axs[0].legend()
axs[0].set_title("Heat 3: spectral loadings (dotted lines: main emission peaks)")
axs[1].set_xlabel("Time (min)"); axs[1].set_ylabel("Score"); axs[1].legend()
axs[1].set_title("Heat 3: smoothed time scores")
fig.tight_layout()
savefig(fig, "q1_components")

fig, ax = plt.subplots(figsize=(6, 4))
for k in HEATS:
    sv = comp_store[k]["sv"][:30]
    ax.semilogy(np.arange(1, 31), sv ** 2 / np.sum(comp_store[k]["sv"] ** 2), "-o", ms=2,
                color="C0" if GROUP[k] == "6B" else "C3", alpha=0.7)
ax.set_xlabel("Component index"); ax.set_ylabel("Explained variance ratio")
ax.set_title("Singular value spectrum (blue: 6B heats, red: 6C heats)")
savefig(fig, "q1_singular_values")

g = inj.groupby("method")[["spike_precision", "spike_recall", "spike_f1", "frame_precision", "frame_recall",
                           "frame_f1"]].mean()
g.columns = ["Spike P", "Spike R", "Spike F1", "Frame P", "Frame R", "Frame F1"]
fig, ax = plt.subplots(figsize=(8, 4))
g.T.plot.bar(ax=ax, rot=0)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title("Synthetic outlier injection test (heats 3 and 8)")
savefig(fig, "q1_detection_pr")

fig, ax = plt.subplots(figsize=(6, 4))
for k, gdf in rank_df.groupby("heat"):
    ax.plot(gdf["rank"], gdf["holdout_rmse_log"], "-o", label=f"Heat {k}")
ax.axvline(R_CHOICE, color="0.4", ls="--", label=f"Chosen r = {R_CHOICE}")
ax.set_xlabel("Rank r"); ax.set_ylabel("Hold-out RMSE (log intensity)"); ax.legend()
ax.set_title("Rank selection by hold-out reconstruction error")
savefig(fig, "q1_rank_selection")
print("Q1 finished")
