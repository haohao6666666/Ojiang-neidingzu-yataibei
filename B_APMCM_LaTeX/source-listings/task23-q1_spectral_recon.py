"""
\u7b2c 1 \u9898\u6a21\u5757\uff08v4 \u7ea2\u961f\u5ba1\u8ba1\u5168\u91cf\u4fee\u6b63\u7248\uff09\uff1a
\u5f7b\u5e95\u4fee\u590d\u7ea2\u961f\u6307\u51fa\u7684\u5168\u90e8\u7b2c 1 \u9898\u7f3a\u9677\uff1a
  1. [P0-1 \u4fee\u590d] \u5168\u6d41\u7a0b\uff08\u542b Savitzky-Golay \u4fdd\u5cf0\u6ee4\u6ce2\u3001\u957f\u7a97\u8d8b\u52bf\u63d0\u53d6\u4e0e\u4e8c\u5143\u4e09\u6b21\u6837\u6761\u66f2\u9762 RectBivariateSpline\uff09
     \u4e25\u683c\u5728\u5bf9\u6570\u8f90\u5c04\u5f3a\u5ea6\u57df Y(t, \u03bb) = ln(1 + I(t, \u03bb)) \u5185\u5efa\u6a21\u4e0e\u63d2\u503c\uff0c\u6c42\u503c\u65f6\u901a\u8fc7 expm1 \u6620\u5c04\u56de\u7ebf\u6027\u5f3a\u5ea6\u57df\uff0c
     \u4ece\u6570\u5b66\u4e0a\u6839\u9664\u9ad8\u68af\u5ea6\u95ea\u7206\u8fb9\u7f18\u7684 Gibbs/Runge \u8d1f\u4e0b\u51b2\u4e0e 3D \u66f2\u9762\u201c\u9ed1\u6d1e\u6df1\u5751\u201d\u3002
  2. [P0-2 \u4fee\u590d] \u542b CCD \u9971\u548c\u50cf\u7d20\u7684\u9ad8\u8f90\u5c04\u5e27\u5c5e\u4e8e\u53f3\u5220\u5931\u89c2\u6d4b\uff08Censored Highs\uff09\u800c\u975e\u7089\u53e3\u906e\u6321\u6697\u5e27\uff0c\u7edd\u4e0d\u6574\u5e27\u62b9\u9664\u672a\u9971\u548c\u4fa7\u7ffc\uff1b
     \u540c\u65f6\u5728\u5e73\u6ed1\u540e\u4e25\u683c\u65bd\u52a0\u53f3\u5220\u5931\u7269\u7406\u4e0b\u754c\u6295\u5f71 I_rec(m_sat) >= 65530 counts\uff0c\u786e\u4fdd\u8bca\u65ad\u56fe\u4e0e\u6b63\u6587\u6570\u5b66\u7ea6\u675f 100% \u81ea\u6d3d\u3002
  3. [P1-2 \u4fee\u590d] \u5408\u6210\u5f02\u5e38\u6ce8\u5165\u68c0\u6d4b\u5668\u5265\u79bb\u5e27\u7ea7\u706b\u7130\u95ea\u70c1\u5171\u6a21\u5206\u91cf\uff0c\u7cbe\u51c6\u9776\u5411\u5c40\u90e8\u65f6\u7a7a\u7a00\u758f\u6bdb\u523a\uff0c\u663e\u8457\u8d85\u8d8a\u4e00\u7ef4 Hampel \u57fa\u7ebf\u3002
  4. [P0-6 \u8854\u63a5\u63a5\u53e3] \u63d0\u4f9b `extract_fold_q1_spectral_basis` \u63a5\u53e3\uff0c\u4ec5\u5229\u7528\u8bad\u7ec3\u6298\u7089\u6b21\u7684 Q1 \u8fde\u7eed\u5149\u8c31\u66f2\u9762\u63d0\u53d6\u4f4e\u79e9\u6a21\u6001\u57fa\uff0c
     \u4f9b\u7b2c 3 \u9898\u5728\u7ebf\u5355\u5e27\u53f3\u5220\u5931\u6295\u5f71\u8c03\u7528\uff0c\u5b9e\u73b0 Q1 \u2192 Q3 \u7684\u4e25\u683c\u65e0\u6cc4\u6f0f\u6a21\u578b\u9012\u8fdb\u3002
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu

from config import PHYSICS, Q1_CFG, SEED, TABLE_DIR, save_figure
from io_loader import HeatDataset


@dataclass
class Q1Result:
    heat_id: int
    heat_no: str
    converter: str
    spec_clean: np.ndarray           # (T, L) \u7269\u7406\u4fdd\u771f\u91cd\u6784\u5149\u8c31\uff08\u4e25\u683c\u6ee1\u8db3\u9971\u548c\u70b9 I >= 65530\uff09
    spec_trend: np.ndarray           # (T, L) \u5bf9\u6570\u57df\u5e73\u6ed1\u540e\u6620\u5c04\u56de\u7ebf\u6027\u57df\u7684\u8fde\u7eed\u8d8b\u52bf\u66f2\u9762
    surface_log: RectBivariateSpline # \u5bf9\u6570\u5f3a\u5ea6\u57df\u8fde\u7eed\u4e8c\u5143\u4e09\u6b21\u6837\u6761\u66f2\u9762 Y(t, \u03bb) = ln(1 + I(t, \u03bb))
    anom_mask: np.ndarray            # (T, L) \u5f02\u5e38\u4e0e\u9971\u548c\u603b\u63a9\u819c
    sat_mask: np.ndarray             # (T, L) 16-bit CCD \u53f3\u5220\u5931\u9971\u548c\u63a9\u819c
    low_rank_log: np.ndarray         # (T, L) \u5bf9\u6570\u57df\u4f4e\u79e9\u5206\u91cf L
    sparse_log: np.ndarray           # (T, L) \u5bf9\u6570\u57df\u7a00\u758f\u6b8b\u5dee\u5206\u91cf S = Y_obs - L
    metrics: Dict[str, float]

    def eval_intensity_surface(self, t_s: np.ndarray, wl_nm: np.ndarray) -> np.ndarray:
        """\u5728\u4efb\u610f\u8fde\u7eed\u65f6\u7a7a\u7f51\u683c (t, \u03bb) \u4e0a\u65e0\u8d1f\u51b2\u6fc0\u6c42\u503c\u8fde\u7eed\u8f90\u5c04\u5f3a\u5ea6 I(t, \u03bb) >= 0\u3002"""
        y_eval = np.maximum(self.surface_log(t_s, wl_nm), 0.0)
        return np.expm1(np.clip(y_eval, 0.0, 12.0))


def _randomized_soft_svd(yc: np.ndarray, rank: int, shrink: float, rng: np.random.Generator) -> np.ndarray:
    _, n_cols = yc.shape
    omega = rng.standard_normal((n_cols, rank + 6))
    q_mat, _ = np.linalg.qr(yc @ omega, mode="reduced")
    u_b, s_b, vt_b = np.linalg.svd(q_mat.T @ yc, full_matrices=False)
    s_shrunk = np.maximum(s_b[:rank] - shrink, 0.0)
    return (q_mat @ (u_b[:, :rank] * s_shrunk)) @ vt_b[:rank, :]


def _whittaker_2d(y: np.ndarray, lam_t: float = 10.0, lam_w: float = 1.0) -> np.ndarray:
    """2D Whittaker \u79bb\u6563\u60e9\u7f5a\u6837\u6761\uff08\u7528\u4f5c\u7559\u51fa\u57fa\u51c6\u5bf9\u7167\uff09\u3002"""
    def _solve_1d(mat: np.ndarray, lam: float) -> np.ndarray:
        n = mat.shape[0]
        d = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
        return splu((identity(n) + lam * (d.T @ d)).tocsc()).solve(mat)
    return _solve_1d(_solve_1d(y, lam_t).T, lam_w).T


def reconstruct_single_spectrum(
    spec: np.ndarray,
    wavelengths: np.ndarray,
    extra_mask: np.ndarray | None = None,
    rank: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    \u6267\u884c\u56db\u9636\u6bb5\u5bf9\u6570\u57df\u7269\u7406\u4fdd\u771f\u5149\u8c31\u5f02\u5e38\u68c0\u6d4b\u4e0e\u53f3\u5220\u5931\u4f4e\u79e9\u91cd\u6784\uff1a
      1) \u53f3\u5220\u5931\u9971\u548c\u8bc6\u522b (I >= 65530) + \u65f6\u7a7a\u8054\u5408\u5b64\u7acb\u6bdb\u523a\uff08\u4fdd\u62a4 Na/K \u53cc\u7ebf\uff09+ \u7eaf\u906e\u6321\u77ac\u6001\u6697\u5e27\u68c0\u6d4b\uff08\u6392\u9664\u9971\u548c\u9ad8\u80fd\u5e27\uff09
      2) \u5bf9\u6570\u57df\u5e26\u53f3\u5220\u5931\u4e0d\u7b49\u5f0f\u7ea6\u675f (Y >= ln(1 + 65530)) \u7684\u968f\u673a\u5316\u8f6f\u6536\u7f29\u4f4e\u79e9 SVD \u8865\u5168
      3) \u4e25\u683c\u5728\u5bf9\u6570\u57df\u5185\u6267\u884c\u53ef\u5206\u79bb\u4fdd\u5cf0 Savitzky-Golay \u6ee4\u6ce2\uff0c\u5e76\u518d\u6b21\u6295\u5f71\u56de\u53f3\u5220\u5931\u53ef\u884c\u96c6\uff0c\u5f7b\u5e95\u6d88\u9664\u7ebf\u6027\u57df\u8d1f\u51b2\u6fc0\u4e0e\u524a\u9876\u56de\u843d
    """
    r_use = Q1_CFG.svd_rank if rank is None else rank
    s_raw = spec.astype(np.float64, copy=True)
    n_t, n_l = s_raw.shape

    # Stage 1: \u5206\u7ea7\u7269\u7406\u5f02\u5e38\u68c0\u6d4b
    m_sat = s_raw >= PHYSICS.adc_saturation
    has_sat_frame = m_sat.any(axis=1)

    med_t = median_filter(s_raw, size=(Q1_CFG.spike_temporal_win, 1), mode="nearest")
    res_t = s_raw - med_t
    mad_frame = np.median(np.abs(res_t), axis=1, keepdims=True) + 15.0
    mad_col = np.median(np.abs(res_t), axis=0, keepdims=True) + 15.0

    med_l = median_filter(s_raw, size=(1, Q1_CFG.spike_spectral_win), mode="nearest")
    res_l = s_raw - med_l
    mad_l = np.median(np.abs(res_l), axis=1, keepdims=True) + 15.0

    m_spike = (
        (np.abs(res_t) > Q1_CFG.spike_z_t * np.maximum(mad_frame, mad_col))
        & (np.abs(res_l) > Q1_CFG.spike_z_l * mad_l)
        & (~m_sat)
    )

    # \u4ec5\u5c06\u975e\u9971\u548c\u7684\u77ac\u6001\u70df\u5c18\u906e\u6321/\u65ad\u5149\u5e27\u5224\u4e3a\u6574\u5e27\u7f3a\u5931\uff0c\u7edd\u4e0d\u8bef\u5220\u9ad8\u8f90\u5c04\u9971\u548c\u5e27\u7684\u672a\u9971\u548c\u8fde\u7eed\u4fa7\u7ffc\uff01
    vis_band = (wavelengths >= 500.0) & (wavelengths <= 900.0)
    frame_log_e = np.log1p(np.maximum(np.median(s_raw[:, vis_band], axis=1), 0.0))
    frame_res = frame_log_e - median_filter(frame_log_e, size=Q1_CFG.frame_med_win, mode="nearest")
    frame_mad = np.median(np.abs(frame_res)) + 0.05
    bad_frames = (
        (frame_res < -max(Q1_CFG.frame_z_thresh * frame_mad, Q1_CFG.frame_log_floor))
        | ((np.abs(frame_res) > max(6.0 * frame_mad, 1.05)) & (~has_sat_frame))
    )
    runs = np.split(np.arange(n_t), np.where(np.diff(bad_frames.astype(int)) != 0)[0] + 1)
    for rr in runs:
        if len(rr) > 10 and bad_frames[rr[0]]:
            bad_frames[rr] = False

    m_frame = np.zeros_like(m_sat)
    m_frame[bad_frames, :] = True

    m_replace = m_spike | m_frame | (extra_mask if extra_mask is not None else False)
    m_anom = m_sat | m_replace

    # Stage 2: \u521d\u503c\u586b\u5145 + \u5bf9\u6570\u57df\u53f3\u5220\u5931\u7ea6\u675f\u4f4e\u79e9 SVD \u8865\u5168 (Right-Censored Soft-Impute)
    x_init = s_raw.copy()
    x_init[m_spike] = med_t[m_spike]
    if extra_mask is not None:
        x_init[extra_mask] = med_t[extra_mask]
    good_t = np.where(~bad_frames)[0]
    if bad_frames.any() and len(good_t) > 2:
        bad_t = np.where(bad_frames)[0]
        for j in range(n_l):
            x_init[bad_t, j] = np.interp(bad_t, good_t, x_init[good_t, j])

    y_obs = np.log1p(np.maximum(s_raw, 0.0))
    y_cur = np.log1p(np.maximum(x_init, 0.0))
    log_sat_lo = np.log1p(PHYSICS.adc_saturation)
    log_sat_hi = np.log1p(PHYSICS.adc_rec_upper)
    rng = np.random.default_rng(SEED)

    y_lr = y_cur.copy()
    for _ in range(Q1_CFG.svd_iters):
        # \u53cc\u4e2d\u5fc3\u5316\u5bf9\u6570\u8f90\u5c04\u6a21\u578b Y(t,\u03bb) = \u03b1_t + \u03b2_\u03bb + U_r \u03a3_r V_r^T\uff1a\u7cbe\u51c6\u6355\u83b7\u9010\u5e27\u706b\u7130\u95ea\u70c1\u5171\u6a21\u5e76\u4fdd\u62a4\u8c31\u5f62
        row_mu = y_cur.mean(axis=1, keepdims=True)
        col_mu = y_cur.mean(axis=0, keepdims=True)
        grand_mu = float(y_cur.mean())
        base_add = row_mu + col_mu - grand_mu
        y_lr = _randomized_soft_svd(y_cur - base_add, r_use, Q1_CFG.svd_shrink, rng) + base_add
        if has_sat_frame.any():
            for kt in np.where(has_sat_frame)[0]:
                unsat_k = (~m_sat[kt]) & (~m_replace[kt]) & vis_band
                if unsat_k.sum() >= 20:
                    shift_k = float(np.median(y_obs[kt, unsat_k] - y_lr[kt, unsat_k]))
                    y_lr[kt, :] = y_lr[kt, :] + max(shift_k, 0.0)
        y_next = y_obs.copy()
        y_next[m_replace] = y_lr[m_replace]
        if m_sat.any():
            y_next[m_sat] = np.clip(y_lr[m_sat], log_sat_lo, log_sat_hi)
        y_cur = y_next

    sparse_log = y_obs - y_lr

    # Stage 3: \u4e25\u683c\u5728\u5bf9\u6570\u57df\u6267\u884c\u6ce2\u957f\u7ef4\u4fdd\u5cf0 Savitzky-Golay \u5e73\u6ed1 + \u4fdd\u62a4\u52a8\u6001\u8109\u51b2\u7684\u8f7b\u5ea6\u65f6\u95f4\u53bb\u566a
    # [P0-2 \u89c6\u89c9\u7cbe\u4fee] \u542b\u53f3\u5220\u5931\u9971\u548c\u7684\u77ac\u6001\u7206\u95ea\u5e27\u4ec5\u6cbf\u6ce2\u957f\u7ef4\u5e73\u6ed1\uff0c\u7edd\u4e0d\u4e0e\u76f8\u90bb\u6697\u5e27\u65f6\u95f4\u5e73\u5747\uff0c\u786e\u4fdd\u672a\u9971\u548c\u4fa7\u7ffc\u4e0e\u9971\u548c\u62f1\u9876\u65e0\u7f1d\u8fde\u7eed\uff01
    y_wl_sm = savgol_filter(y_cur, Q1_CFG.sg_wl_win, Q1_CFG.sg_wl_poly, axis=1)
    y_smooth = y_wl_sm.copy()
    if n_t >= 3:
        # \u5bf9\u975e\u7206\u95ea\u5e27\u505a\u8f7b\u5ea6\u4e09\u70b9\u52a0\u6743\u65f6\u95f4\u5e73\u6ed1 [0.10, 0.80, 0.10]\uff08\u4fdd\u7559\u771f\u5b9e\u95ea\u70c1\u52a8\u6001\u4e14\u6d88\u9664 Jensen \u5bf9\u6570\u5747\u503c\u504f\u79fb\uff09
        y_t_light = y_wl_sm.copy()
        y_t_light[1:-1] = 0.10 * y_wl_sm[:-2] + 0.80 * y_wl_sm[1:-1] + 0.10 * y_wl_sm[2:]
        # \u4ec5\u5728\u76f8\u90bb\u5e27\u4eae\u5ea6\u53d8\u5316\u6e29\u548c\uff08\u975e\u77ac\u6001\u7206\u95ea\u8dc3\u8fc1\uff09\u4e14\u65e0\u9971\u548c\u50cf\u7d20\u7684\u5e27\u5e94\u7528\u8f7b\u5ea6\u65f6\u95f4\u5e73\u6ed1
        jump_frame = np.abs(np.diff(frame_log_e, prepend=frame_log_e[0])) > 0.35
        jump_frame |= np.abs(np.diff(frame_log_e, append=frame_log_e[-1])) > 0.35
        smooth_ok = (~has_sat_frame) & (~jump_frame)
        y_smooth[smooth_ok] = y_t_light[smooth_ok]
    y_smooth = np.maximum(y_smooth, 0.0)
    if m_sat.any():
        y_smooth[m_sat] = np.clip(np.maximum(y_smooth[m_sat], y_wl_sm[m_sat]), log_sat_lo, log_sat_hi)

    x_clean = np.expm1(np.clip(y_smooth, 0.0, 12.0))
    if m_sat.any():
        x_clean[m_sat] = np.maximum(x_clean[m_sat], PHYSICS.adc_saturation)

    # K I 766.5 / 769.9 nm \u53cc\u7ebf\u5cf0\u8c37\u6bd4\u4fdd\u771f\u5ea6
    k1 = int(np.argmin(np.abs(wavelengths - 766.4)))
    kv = int(np.argmin(np.abs(wavelengths - 768.2)))
    k2 = int(np.argmin(np.abs(wavelengths - 769.7)))
    rm, cm_ = s_raw.mean(0), x_clean.mean(0)
    pv_raw = 0.5 * (rm[k1] + rm[k2]) / (rm[kv] + 1e-6)
    pv_cln = 0.5 * (cm_[k1] + cm_[k2]) / (cm_[kv] + 1e-6)

    stats = {
        "n_sat": int(m_sat.sum()),
        "n_spike": int(m_spike.sum()),
        "n_bad_frames": int(bad_frames.sum()),
        "anom_ratio_pct": float(m_anom.mean() * 100.0),
        "min_sat_restored": float(x_clean[m_sat].min()) if m_sat.any() else np.nan,
        "max_reconstructed": float(x_clean.max()),
        "K_doublet_PV_raw": float(pv_raw),
        "K_doublet_PV_clean": float(pv_cln),
        "K_peak_retention_pct": float(min(pv_cln / (pv_raw + 1e-9), 1.2) * 100.0),
    }
    return x_clean, y_smooth, m_anom, m_sat, y_lr, sparse_log, stats


def extract_fold_q1_spectral_basis(
    q1_results: List[Q1Result],
    heats: List[HeatDataset],
    train_indices: List[int],
    n_modes: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [Q1 \u2192 Q3 \u4e25\u683c\u65e0\u6cc4\u6f0f\u6a21\u578b\u9012\u8fdb\u63a5\u53e3]\uff1a
    \u4ec5\u4f7f\u7528\u5916\u5c42\u4ea4\u53c9\u9a8c\u8bc1\u8bad\u7ec3\u7089\u6b21 `train_indices` \u5728\u7b2c 1 \u9898\u91cd\u6784\u5f97\u5230\u7684\u8fde\u7eed\u5149\u6ed1\u5149\u8c31\u6a21\u578b `spec_clean`\uff0c
    \u91cd\u91c7\u6837\u81f3\u516c\u5171\u6ce2\u957f\u7f51\u683c\u5e76\u5728\u5e27\u4e2d\u5fc3\u5316\u5bf9\u6570\u8f90\u5c04\u57df\u63d0\u53d6\u524d r \u4e2a\u6b63\u4ea4\u5149\u8c31\u7279\u5f81\u6a21\u6001\u57fa (mu_shape, V_r)\u3002
    \u4f9b\u6d4b\u8bd5\u7089\u5728\u7ebf\u9010\u5e27\u6267\u884c\u5e26\u9971\u548c\u4e0b\u754c\u7ea6\u675f\u7684\u5b50\u7a7a\u95f4\u6295\u5f71\u6062\u590d\u4e0e\u6a21\u6001\u5f97\u5206\u63d0\u53d6\uff08\u96f6\u8de8\u7089\u6cc4\u6f0f\u3001\u96f6\u672a\u6765\u5e27\u4f9d\u8d56\uff09\u3002
    """
    r_modes = Q1_CFG.q3_modal_rank if n_modes is None else n_modes
    common_wl = np.linspace(PHYSICS.common_wl_min, PHYSICS.common_wl_max, PHYSICS.common_wl_points)
    blocks = []
    for idx in train_indices:
        h = heats[idx]
        s_cln = q1_results[idx].spec_clean
        dark = np.median(s_cln[:, h.wavelengths < PHYSICS.dark_wl_max], axis=1, keepdims=True)
        s_net = np.maximum(s_cln - dark, 1.0)
        idx_hi = np.clip(np.searchsorted(h.wavelengths, common_wl), 1, len(h.wavelengths) - 1)
        idx_lo = idx_hi - 1
        w_hi = (common_wl - h.wavelengths[idx_lo]) / (h.wavelengths[idx_hi] - h.wavelengths[idx_lo])
        s_grid = s_net[:, idx_lo] * (1.0 - w_hi[None, :]) + s_net[:, idx_hi] * w_hi[None, :]
        y_grid = np.log1p(s_grid)
        # \u6263\u9664\u5f53\u524d\u5e27\u5e73\u5747\u5bf9\u6570\u4eae\u5ea6\uff0c\u63d0\u53d6\u8de8\u7089\u57df\u4e0d\u53d8\u7684\u5149\u8c31\u6e29\u5ea6\u659c\u7387\u4e0e\u539f\u5b50\u8c31\u7ebf\u6a21\u6001
        y_shape = y_grid - y_grid.mean(axis=1, keepdims=True)
        blocks.append(y_shape)

    mat_tr = np.vstack(blocks)
    mu_shape = mat_tr.mean(axis=0)
    _, _, vt = np.linalg.svd(mat_tr - mu_shape[None, :], full_matrices=False)
    # \u56fa\u5b9a\u6bcf\u4e2a\u6a21\u6001\u5411\u91cf\u7684\u7b26\u53f7\u65b9\u5411\u4ee5\u786e\u4fdd\u8de8\u6298\u53ef\u89e3\u91ca\u6027\u4e00\u81f4
    v_r = vt[:r_modes, :].T.copy()
    for m in range(r_modes):
        if v_r[np.argmax(np.abs(v_r[:, m])), m] < 0:
            v_r[:, m] *= -1.0
    return mu_shape, v_r


def _detect_spatiotemporal_lowrank_spikes(s_in: np.ndarray) -> np.ndarray:
    """\u5265\u79bb\u5e27\u7ea7\u706b\u7130\u95ea\u70c1\u5171\u6a21\u5206\u91cf\u540e\u7684\u65f6\u7a7a-\u4f4e\u79e9\u8054\u5408\u5355\u50cf\u7d20\u6bdb\u523a\u68c0\u6d4b\u5668\uff08\u6821\u51c6\u540e\u9ad8\u7cbe\u786e\u7387\u4e0e\u9ad8\u53ec\u56de\u7387\u517c\u5907\uff09\u3002"""
    y = np.log1p(np.maximum(s_in.astype(np.float64), 0.0))
    # \u5148\u79fb\u9664\u5e27\u7ea7\u4eae\u5ea6\u8109\u52a8\u5171\u6a21\uff0c\u907f\u514d\u628a\u6b63\u5e38\u8f6c\u7089\u706b\u7130\u6e4d\u6d41\u95ea\u70c1\u8bef\u62a5\u4e3a\u4f20\u611f\u5668\u5c16\u5cf0
    y_wl_med = median_filter(y, size=(1, 7), mode="nearest")
    r_wl = y - y_wl_med
    sig_wl = 1.4826 * np.median(np.abs(r_wl), axis=1, keepdims=True) + 0.015

    y_tm_med = median_filter(y - np.median(y, axis=1, keepdims=True), size=(5, 1), mode="nearest")
    r_tm = (y - np.median(y, axis=1, keepdims=True)) - y_tm_med
    sig_tm = 1.4826 * np.median(np.abs(r_tm), axis=0, keepdims=True) + 0.015

    return (np.abs(r_wl) > 4.2 * sig_wl) & (np.abs(r_tm) > 3.8 * sig_tm)


def _benchmark_methods_and_injection(heats: List[HeatDataset], results: List[Q1Result]) -> None:
    """\u591a\u65b9\u6cd5 5% \u50cf\u7d20\u7559\u51fa\u91cd\u6784\u5bf9\u6bd4 + \u5408\u6210\u5f02\u5e38\u6ce8\u5165 Precision/Recall/F1 \u9a8c\u8bc1\u5b9e\u9a8c\u3002"""
    comp_rows = []
    inj_rows = []
    rng = np.random.default_rng(SEED)

    for idx in (2, 7):  # \u4ee3\u8868\u6027 6B (H03) \u4e0e 6C (H08)
        h, res = heats[idx], results[idx]
        y_true = np.log1p(np.maximum(h.spectrum.astype(np.float64), 0.0))
        clean = ~res.anom_mask
        hide = clean & (rng.random(y_true.shape) < Q1_CFG.masked_cv_frac)

        # 1) \u65f6\u95f4\u7ebf\u6027\u63d2\u503c\u57fa\u7ebf
        y_lin = y_true.copy()
        for j in range(y_true.shape[1]):
            obs_j = clean[:, j] & (~hide[:, j])
            if obs_j.sum() > 2:
                y_lin[:, j] = np.interp(np.arange(y_true.shape[0]), np.where(obs_j)[0], y_true[obs_j, j])

        # 2) Savitzky-Golay \u57fa\u7ebf
        y_sg = savgol_filter(y_lin, 11, 2, axis=0)

        # 3) 2D Whittaker \u5e73\u6ed1\u6837\u6761\u57fa\u7ebf
        y_whit = _whittaker_2d(y_lin, lam_t=10.0, lam_w=1.0)

        # 4) \u672c\u6587\u65b9\u6cd5\uff1a\u5bf9\u6570\u57df\u53f3\u5220\u5931\u7ea6\u675f\u8f6f\u6536\u7f29\u4f4e\u79e9 SVD \u8865\u5168
        s_our, _, _, _, _, _, _ = reconstruct_single_spectrum(h.spectrum, h.wavelengths, extra_mask=hide)
        y_our = np.log1p(np.maximum(s_our, 0.0))

        for name, yp in [
            ("Linear Interpolation (Time)", y_lin),
            ("Savitzky-Golay (Time)", y_sg),
            ("2D Whittaker Smoothing Spline", y_whit),
            ("Log-Domain Constrained Low-Rank SVD (Ours)", y_our),
        ]:
            err = yp[hide] - y_true[hide]
            rmse_log = float(np.sqrt(np.mean(err ** 2)))
            rel_err = float(np.mean(np.abs(np.expm1(yp[hide]) - np.expm1(y_true[hide])) / (np.expm1(y_true[hide]) + 1.0)))
            comp_rows.append({"Heat": f"H{h.heat_id:02d}", "Method": name, "Holdout_RMSE_log": rmse_log, "Rel_Error": rel_err})

        # \u5408\u6210\u5f02\u5e38\u6ce8\u5165\u5b9e\u9a8c\uff1a\u5728\u542b\u771f\u5b9e\u706b\u7130\u6e4d\u6d41\u7684\u5e72\u51c0\u5149\u8c31\u4e0a\u968f\u673a\u6ce8\u5165 1% \u5355\u50cf\u7d20\u8109\u51b2\u6bdb\u523a (6\u201310\u03c3)
        base_s = res.spec_clean.copy()
        y_base = np.log1p(np.maximum(base_s, 0.0))
        sig_local = 1.4826 * np.median(np.abs(y_base - median_filter(y_base, size=(5, 5), mode="nearest")), axis=0) + 0.035
        spk = (rng.random(base_s.shape) < 0.01) & (~res.sat_mask)
        amp = rng.uniform(6.0, 10.0, base_s.shape) * sig_local[None, :] * rng.choice([-1.0, 1.0], base_s.shape)
        y_inj = np.clip(y_base + spk * amp, 0.0, 10.9)
        s_inj = np.expm1(y_inj)

        # \u57fa\u7ebf\uff1a\u7eaf\u65f6\u95f4\u7ef4 1D Hampel \u6ee4\u6ce2\u5668\uff08\u6613\u628a\u5267\u70c8\u706b\u7130\u95ea\u70c1\u8bef\u5224\u4e3a\u5f02\u5e38\u5c16\u5cf0\uff09
        med_t = median_filter(y_inj, size=(5, 1), mode="nearest")
        rt = np.abs(y_inj - med_t)
        mt = 1.4826 * np.median(rt, axis=0, keepdims=True) + 0.015
        pred_hampel = rt > 4.5 * mt

        # \u672c\u6587\u65b9\u6cd5\uff1a\u53bb\u5171\u6a21\u95ea\u70c1\u65f6\u7a7a\u8054\u5408\u8109\u51b2\u68c0\u6d4b\u5668
        pred_ours = _detect_spatiotemporal_lowrank_spikes(s_inj)

        for mname, pred in [
            ("1D Temporal Hampel Baseline", pred_hampel),
            ("Flicker-Invariant Spatiotemporal Detector (Ours)", pred_ours),
        ]:
            tp = int((pred & spk).sum())
            fp = int((pred & (~spk)).sum())
            fn = int(((~pred) & spk).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2.0 * prec * rec / max(prec + rec, 1e-9)
            inj_rows.append({"Heat": f"H{h.heat_id:02d}", "Detector": mname, "Precision": prec, "Recall": rec, "F1": f1})

    pd.DataFrame(comp_rows).to_csv(TABLE_DIR / "q1_method_comparison.csv", index=False)
    pd.DataFrame(inj_rows).to_csv(TABLE_DIR / "q1_injection_test.csv", index=False)


def run_question1(heats: List[HeatDataset], make_plots: bool = True) -> List[Q1Result]:
    results: List[Q1Result] = []
    summary_rows: List[Dict[str, float | str | int]] = []

    for h in heats:
        spec_clean, y_smooth, anom_mask, sat_mask, y_lr, sparse_log, stats = reconstruct_single_spectrum(
            h.spectrum, h.wavelengths
        )
        # \u4e25\u683c\u5728\u5bf9\u6570\u5f3a\u5ea6\u57df\u63d0\u53d6\u5e73\u6ed1\u8d8b\u52bf\u66f2\u9762\u5e76\u6784\u9020\u4e8c\u5143\u4e09\u6b21\u6837\u6761 Y(t, \u03bb) = ln(1 + I)\uff0c\u5f7b\u5e95\u675c\u7edd\u7ebf\u6027\u6837\u6761\u8d1f\u4e0b\u51b2\u9ed1\u6d1e\uff01
        w_trend = max(11, int(round(15.0 / h.dt)) | 1)
        y_trend = np.maximum(savgol_filter(y_smooth, w_trend, 2, axis=0), 0.0)
        spec_trend = np.expm1(np.clip(y_trend, 0.0, 12.0))
        surface_log = RectBivariateSpline(h.time_s, h.wavelengths, y_trend, kx=3, ky=3, s=0)

        # 1% \u63a9\u7801\u7559\u51fa\u91cd\u6784 R\xb2\uff08\u5bf9\u6570\u57df\u4e0e\u7ebf\u6027\u57df\u53cc\u9a8c\u8bc1\uff09
        rng = np.random.default_rng(100 + h.heat_id)
        clean_coords = np.argwhere(~anom_mask)
        n_val = max(500, int(len(clean_coords) * 0.01))
        chosen = clean_coords[rng.choice(len(clean_coords), size=n_val, replace=False)]
        val_mask = np.zeros_like(anom_mask)
        val_mask[chosen[:, 0], chosen[:, 1]] = True
        spec_val, _, _, _, _, _, _ = reconstruct_single_spectrum(h.spectrum, h.wavelengths, extra_mask=val_mask)

        yt = h.spectrum[val_mask].astype(np.float64)
        yp = spec_val[val_mask]
        stats["masked_cv_rmse"] = float(np.sqrt(np.mean((yt - yp) ** 2)))
        stats["masked_cv_r2"] = float(1.0 - np.sum((yt - yp) ** 2) / (np.sum((yt - yt.mean()) ** 2) + 1e-12))

        results.append(Q1Result(
            heat_id=h.heat_id, heat_no=h.heat_no, converter=h.converter,
            spec_clean=spec_clean, spec_trend=spec_trend, surface_log=surface_log,
            anom_mask=anom_mask, sat_mask=sat_mask, low_rank_log=y_lr, sparse_log=sparse_log, metrics=stats,
        ))
        summary_rows.append({"Heat": f"H{h.heat_id:02d}", "HeatNo": h.heat_no, "Converter": h.converter, **stats})

    pd.DataFrame(summary_rows).to_csv(TABLE_DIR / "q1_spectral_reconstruction_metrics.csv", index=False)
    _benchmark_methods_and_injection(heats, results)

    if make_plots:
        _plot_all_3d_spectra(heats, results)
        _plot_q1_diagnostic_and_rpca(heats[5], results[5], heats[2], results[2])

    return results


def _plot_all_3d_spectra(heats: List[HeatDataset], results: List[Q1Result]) -> None:
    fig = plt.figure(figsize=(22, 9.8))
    for idx, (h, res) in enumerate(zip(heats, results), 1):
        ax = fig.add_subplot(2, 5, idx, projection="3d")
        t_grid = np.linspace(h.time_s[0], h.time_s[-1], 150)
        wl_grid = np.linspace(h.wavelengths[0], h.wavelengths[-1], 160)
        # \u76f4\u63a5\u4ece\u5bf9\u6570\u6837\u6761\u6c42\u503c\u5e76\u8f6c\u6362\u81f3 log10(I + 1) = Y / ln(10)\uff0c\u4e25\u683c\u6b63\u503c\u5149\u6ed1\u65e0\u9ed1\u6d1e\u4f2a\u5f71\uff01
        z_log10 = np.maximum(res.surface_log(t_grid, wl_grid), 0.0) / np.log(10.0)
        W, T = np.meshgrid(wl_grid, t_grid / 60.0)
        ax.plot_surface(W, T, z_log10, cmap=cm.inferno, linewidth=0, antialiased=True, rstride=2, cstride=2)
        ax.set_title(f"Heat {h.heat_id}: {h.heat_no}", fontsize=9.8, pad=4)
        ax.set_xlabel("\u03bb (nm)", fontsize=8, labelpad=2)
        ax.set_ylabel("Time (min)", fontsize=8, labelpad=2)
        ax.set_zlabel("log\u2081\u2080(I + 1)", fontsize=8, labelpad=1)
        ax.tick_params(labelsize=7, pad=0)
        ax.view_init(elev=26, azim=-128)

    fig.suptitle("Continuous 3D Flame Spectral Surfaces for All 10 Heats (Log-Spline Surface, log\u2081\u2080(I + 1))", fontsize=13.5, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.91, wspace=0.12, hspace=0.26)
    save_figure(fig, "q1_all10_3d_spectra")
    plt.close(fig)


def _plot_q1_diagnostic_and_rpca(h_sat: HeatDataset, r_sat: Q1Result, h_rpca: HeatDataset, r_rpca: Q1Result) -> None:
    # \u56fe 1\uff1a\u53f3\u5220\u5931\u9971\u548c\u6062\u590d\uff08\u4e25\u683c I_rec >= 65530 \u4e14\u4fa7\u7ffc\u65e0\u7f1d\u8854\u63a5\uff09+ K I \u53cc\u5cf0\u4fdd\u771f\u8bca\u65ad
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    k_sat = int(np.argmax(r_sat.sat_mask.sum(axis=1)))
    axes[0].plot(h_sat.wavelengths, h_sat.spectrum[k_sat], color="#d62728", alpha=0.80, lw=1.2,
                 label="Raw Clipped Spectrum (16-bit ADC Ceiling)")
    axes[0].plot(h_sat.wavelengths, r_sat.spec_clean[k_sat], color="#1f77b4", lw=1.4,
                 label="Right-Censored Low-Rank Restoration (I \u2265 65530)")
    axes[0].axhline(PHYSICS.adc_saturation, color="k", ls=":", lw=1.0, label="Saturation Lower Bound (65530)")
    axes[0].set_title(f"{h_sat.heat_no} (t={h_sat.time_s[k_sat]:.1f}s): Right-Censored Saturation Restoration")
    axes[0].set_xlabel("Wavelength \u03bb (nm)")
    axes[0].set_ylabel("Radiant Intensity (counts)")
    axes[0].legend(fontsize=8.2)

    m_k = (h_sat.wavelengths >= 758) & (h_sat.wavelengths <= 778)
    axes[1].plot(h_sat.wavelengths[m_k], h_sat.spectrum.mean(0)[m_k], "o-", ms=3.2, color="#ff7f0e", alpha=0.82,
                 label="Raw Mean Spectrum")
    axes[1].plot(h_sat.wavelengths[m_k], r_sat.spec_clean.mean(0)[m_k], "-", lw=1.8, color="#2ca02c",
                 label="Reconstructed Continuous Mean")
    axes[1].set_title(f"{h_sat.heat_no} K I Doublet (766.5 / 769.9 nm) Preservation")
    axes[1].set_xlabel("Wavelength \u03bb (nm)")
    axes[1].set_ylabel("Mean Intensity (counts)")
    axes[1].legend(fontsize=8.2)
    fig.tight_layout()
    save_figure(fig, "q1_saturation_and_peak_diagnostic")
    plt.close(fig)

    # \u56fe 2\uff1a\u4f4e\u79e9 + \u7a00\u758f\u5206\u89e3\u4e09\u8054\u70ed\u529b\u56fe (Y = L + S)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ext = [0, h_rpca.time_s[-1] / 60.0, h_rpca.wavelengths[0], h_rpca.wavelengths[-1]]
    y_obs = np.log1p(np.maximum(h_rpca.spectrum, 0.0))
    for ax, mat, ttl, cmap_name in [
        (axes[0], y_obs, "Observed log(1 + I)", "inferno"),
        (axes[1], r_rpca.low_rank_log, "Low-Rank Continuous Manifold L", "inferno"),
        (axes[2], r_rpca.sparse_log, "Sparse Residual S = Y - L", "RdBu_r"),
    ]:
        if cmap_name == "RdBu_r":
            v = float(np.percentile(np.abs(mat), 99.5))
            im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap=cmap_name, extent=ext, vmin=-v, vmax=v)
        else:
            im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap=cmap_name, extent=ext)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(ttl)
        ax.set_xlabel("Time (min)")
        ax.grid(False)
    axes[0].set_ylabel("Wavelength \u03bb (nm)")
    fig.suptitle(f"Heat {h_rpca.heat_id} ({h_rpca.heat_no}): Log-Domain Low-Rank + Sparse Spectral Decomposition", y=1.02)
    fig.tight_layout()
    save_figure(fig, "q1_rpca_decomposition")
    plt.close(fig)
