"""
\u7b2c 3 \u9898\u6a21\u5757\uff08v4 \u7ea2\u961f\u5ba1\u8ba1\u5168\u91cf\u4fee\u6b63\u7248\uff09\uff1a
\u5f7b\u5e95\u89e3\u51b3\u7ea2\u961f\u62a5\u544a\u6307\u51fa\u7684\u5168\u90e8\u65b9\u6cd5\u8bba\u3001\u6a21\u578b\u8854\u63a5\u3001\u5d4c\u5957\u6cc4\u6f0f\u4e0e\u65b0\u7248 H01 \u5916\u63a8\u77ed\u677f\uff1a
  1. [P0-6 Q1\u2192Q3 \u4e25\u683c\u65e0\u6cc4\u6f0f\u6a21\u578b\u9012\u8fdb]\uff1a
     \u5728\u6bcf\u4e2a\u5916\u5c42\u7559\u4e00\u7089\uff08LOHO\uff09\u6298 i \u5185\uff0c\u4ec5\u4ece\u5176\u4f59 9 \u4e2a\u8bad\u7ec3\u7089\u5728\u7b2c 1 \u9898\u91cd\u6784\u5f97\u5230\u7684\u8fde\u7eed\u5149\u6ed1\u5149\u8c31\u6a21\u578b\u63d0\u53d6\u4f4e\u79e9\u6a21\u6001\u57fa
     (\u03bc^{(-i)}, V_r^{(-i)})\uff1b\u5728\u7ebf\u90e8\u7f72\u4e0e\u6d4b\u8bd5\u65f6\uff0c\u9010\u5e27\u5728\u672a\u9971\u548c\u901a\u9053 \u03a9_t \u4e0a\u6267\u884c\u5e26\u53f3\u5220\u5931\u4e0b\u754c\u7ea6\u675f\u7684\u5b50\u7a7a\u95f4\u6295\u5f71\u6062\u590d\u5e76\u63d0\u53d6
     Q1 \u8fde\u7eed\u6a21\u6001\u5f97\u5206 a_t \u2208 R^r \u2014\u2014 \u65e2\u771f\u6b63\u5c65\u884c\u9898\u9762\u201c\u4f7f\u7528\u7b2c(1)(2)\u95ee\u8fde\u7eed\u4fe1\u53f7\u6a21\u578b\u201d\u7684\u9012\u8fdb\u8981\u6c42\uff0c\u53c8\u4e25\u683c\u4fdd\u6301\u672a\u6765\u5e27\u7be1\u6539\u8bef\u5dee == 0.00e+00\uff01
  2. [P0-4 & P0-6 \u7ebf\u6027\u8d8b\u52bf-\u6811\u6b8b\u5dee\u6df7\u5408 + \u56e0\u679c\u8f6f\u95e8\u63a7\u591a\u7089\u96c6\u6210]\uff1a
     \u79fb\u9664\u5f15\u5165\u8de8\u6c14\u4f53\u566a\u58f0\u4e32\u6270\u7684\u4f2a\u62c9\u683c\u6717\u65e5\u6295\u5f71\uff0c\u6539\u7528\u201c\u6709\u754c\u7269\u7406\u57fa Ridge \u7ebf\u6027\u8d8b\u52bf\u5916\u63a8\u5934 + \u591a\u6df1\u5ea6 XGBoost \u6b8b\u5dee\u6811\u201d\u6df7\u5408\u6a21\u578b\uff0c
     \u5e76\u5bf9\u6bd4\uff1a(a) \u6309\u8f6c\u7089\u53f7\u786c\u5206\u7ec4 (6B/6C)\u3001(b) \u57fa\u4e8e\u6298\u5185\u524d 60 s \u56e0\u679c\u7279\u5f81\u7684 Ward \u5c42\u6b21\u805a\u7c7b\u5206\u7ec4\u3001(c) \u5168\u7089\u6c60\u5316\u6a21\u578b\u3001
     (d) \u5168\u7089\u6df7\u5408\u4e3b\u5e72 + \u57fa\u4e8e\u524d 60 s \u56e0\u679c\u76f8\u4f3c\u5ea6\u6838\u6743\u91cd\u7684\u4e13\u5bb6\u8f6f\u95e8\u63a7\u96c6\u6210\uff08Combined_SoftGated_Hybrid\uff09\uff0c\u6b63\u9762\u56de\u5e94\u591a\u7089\u7ec4\u5408\u547d\u9898\u3002
  3. [P0-6 \u5f7b\u5e95\u6d88\u9664\u5185\u5c42\u5d4c\u5957\u6cc4\u6f0f + \u5c40\u90e8\u5c3a\u5ea6\u81ea\u9002\u5e94\u5171\u5f62\u9884\u6d4b]\uff1a
     \u5728\u5916\u5c42\u6298 i \u7684 9 \u4e2a\u8bad\u7ec3\u7089\u5185\u90e8\u6267\u884c\u4e25\u683c\u7684 8-vs-1 \u5185\u5c42\u4ea4\u53c9\u9a8c\u8bc1\uff08Inner-OOF\uff09\uff0c\u4ec5\u7528\u5185\u5c42 OOF \u9884\u6d4b\u6807\u5b9a\u6df7\u5408\u6743\u91cd\u3001
     Smith \u4e92\u8865\u89c2\u6d4b\u5668\u589e\u76ca g^{(-i)} \u4e0e\u5c40\u90e8\u5c3a\u5ea6\u5f52\u4e00\u5316\u5171\u5f62\u5206\u4f4d\u6570 q_{0.90}^{(-i)}\uff0c\u7edd\u4e0d\u89e6\u78b0\u5305\u542b\u6d4b\u8bd5\u7089\u7684\u5916\u5c42\u9884\u6d4b\u6216\u5168\u5c40\u5e38\u6570\uff01
  4. [P1-1 \u589e\u8865\u6ede\u540e\u786c\u4ef6\u5206\u6790\u4eea\u6301\u4e45\u6027\u57fa\u7ebf]\uff1a
     \u5728\u5168\u90e8\u91c7\u6837\u95f4\u9694\u4e0b\u52a0\u5165\u672a\u7ecf\u65f6\u6ede\u8865\u507f\u4e0e\u5149\u8c31\u878d\u5408\u7684 `Naive_Lagged_Sensor` \u786c\u4ef6\u76f4\u8bfb\u57fa\u7ebf\uff0c\u91cf\u5316\u7eaf\u786c\u4ef6\u5206\u6790\u4eea\u7684\u4e25\u91cd\u6ede\u540e\u5931\u771f\u3002
  5. [P0-5, P1-4, P1-5]\uff1a
     \u56e0\u679c\u6c27\u67aa\u6d41\u91cf\u540c\u6b65\u8fc7\u6ee4 PLC \u8865\u8d26\u8df3\u53d8\uff1b\u805a\u7c7b\u6811\u72b6\u56fe\u6539\u7528\u4e25\u683c\u524d 60 s \u56e0\u679c\u63cf\u8ff0\u7b26\u5e76\u62d3\u5bbd\u753b\u5e03\u6d88\u9664\u6807\u7b7e\u91cd\u53e0\u3002
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config import PHYSICS, Q1_CFG, Q3_CFG, SEED, TABLE_DIR, save_figure
from io_loader import HeatDataset
from q1_spectral_recon import Q1Result, extract_fold_q1_spectral_basis
from q2_gas_delay import PFRDelayParams, Q2HeatRecon, Q2Summary, fit_pfr_kinetics


@dataclass
class CausalOnlineStreams:
    """100% \u7eaf\u5355\u4fa7\u56e0\u679c\u5728\u7ebf\u6570\u636e\u6d41\uff08\u878d\u5408\u6298\u5185\u8bad\u7ec3\u4fa7\u63d0\u53d6\u7684 Q1 \u8fde\u7eed\u5149\u8c31\u6a21\u6001\u57fa\u5355\u5e27\u6295\u5f71\uff0c\u96f6\u672a\u6765\u5e27\u4f9d\u8d56\uff09\u3002"""
    heat_id: int
    heat_no: str
    converter: str
    dt: float
    time_s: np.ndarray
    spec_prob: np.ndarray        # (T, 127) \u5355\u5e27\u53f3\u5220\u5931 Q1 \u6a21\u6001\u6295\u5f71\u6062\u590d\u540e\u7684\u76f8\u5bf9\u6982\u7387\u8c31
    q1_modal_scores: np.ndarray  # (T, r) \u5728\u7ebf\u6295\u5f71\u5f97\u5230\u7684\u7b2c 1 \u9898\u4f4e\u79e9\u8fde\u7eed\u5149\u8c31\u6a21\u6001\u5f97\u5206
    x_proc: np.ndarray           # M0: \u7eaf\u5355\u4fa7\u56e0\u679c\u5de5\u827a\u7279\u5f81
    x_spec: np.ndarray           # M1: \u7eaf\u5355\u4fa7\u56e0\u679c\u5149\u8c31\uff08\u8f90\u5c04\u7279\u5f81 + Q1 \u6a21\u6001\u5f97\u5206\uff09
    x_m3: np.ndarray             # M3: \u5149\u8c31 + \u5de5\u827a\u7279\u5f81
    x_m4: np.ndarray             # M4: M3 + \u524d 60 s \u4e25\u683c\u56e0\u679c\u7089\u6b21\u57fa\u7ebf\u5f52\u4e00\u5316\u7279\u5f81
    x_lin: np.ndarray            # \u4f9b\u7ebf\u6027\u5916\u63a8\u5934\u4f7f\u7528\u7684\u6709\u754c\u5e73\u6ed1\u7269\u7406\u57fa\u7279\u5f81
    desc_causal60: np.ndarray    # (D_desc,) \u4e25\u683c\u4ec5\u7531\u524d 60 s \u6570\u636e\u6784\u6210\u7684\u56e0\u679c\u7089\u6b21\u5de5\u51b5\u6307\u7eb9\uff08\u4f9b\u805a\u7c7b\u4e0e\u8f6f\u95e8\u63a7\u4f7f\u7528\uff09
    o2_causal: np.ndarray        # \u56e0\u679c\u5355\u8c03\u7d2f\u79ef\u8017\u6c27\u91cf
    q_causal: np.ndarray         # \u53bb\u8865\u8d26\u8df3\u53d8\u540e\u7684\u56e0\u679c 24 s \u540e\u5411\u659c\u7387\u6d41\u91cf
    obs_co2_causal: np.ndarray   # \u56e0\u679c\u5355\u4fa7\u5e73\u6ed1\u540e\u7684\u5206\u6790\u4eea\u5f53\u524d\u6ede\u540e\u8bfb\u6570
    obs_co_causal: np.ndarray
    raw_co2_zoh: np.ndarray      # \u539f\u59cb\u5206\u6790\u4eea\u5f53\u524d\u65f6\u523b\u96f6\u9636\u4fdd\u6301\u8bfb\u6570\uff08\u4f9b Naive_Lagged_Sensor \u57fa\u7ebf\u8bc4\u4f30\uff09
    raw_co_zoh: np.ndarray


def _causal_ema(arr: np.ndarray, tau_s: float, dt: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-dt / tau_s)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for k in range(1, len(arr)):
        out[k] = alpha * arr[k] + (1.0 - alpha) * out[k - 1]
    return out


def compute_strictly_causal_streams(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    o2_raw: np.ndarray,
    co2_raw: np.ndarray,
    co_raw: np.ndarray,
    dt: float,
    heat_id: int = 0,
    heat_no: str = "",
    q1_basis: Tuple[np.ndarray, np.ndarray] | None = None,
    o2_scale_m3: float | None = None,
) -> CausalOnlineStreams:
    """
    \u4e25\u683c\u9010\u5e27/\u5355\u4fa7\u56e0\u679c\u5728\u7ebf\u5904\u7406\u7ba1\u7ebf\uff1a
      - \u82e5\u4f20\u5165\u6298\u5185\u8bad\u7ec3\u7089\u63d0\u53d6\u7684 Q1 \u8fde\u7eed\u5149\u8c31\u6a21\u6001\u57fa `q1_basis = (mu_shape, V_r)`\uff0c\u5219\u5bf9\u5f53\u524d\u5e27\u672a\u9971\u548c\u50cf\u7d20\u6267\u884c
        \u5e26\u6b63\u5219\u5316\u7684\u5355\u5e27\u5b50\u7a7a\u95f4\u6700\u5c0f\u4e8c\u4e58\u6295\u5f71\u6062\u590d\u9971\u548c\u6ce2\u6bb5\u5e76\u8f93\u51fa Q1 \u8fde\u7eed\u6a21\u6001\u5f97\u5206\uff1b
      - \u7edd\u4e0d\u4f7f\u7528\u4efb\u4f55\u53cc\u4fa7\u7a97\u53e3\u3001\u6d4b\u8bd5\u7089\u6574\u7089\u7edf\u8ba1\u91cf\u6216\u672a\u6765\u65f6\u523b\u6570\u636e\uff01
    """
    s_raw = spectrum.astype(np.float64, copy=False)
    n, _ = s_raw.shape
    t = np.arange(n, dtype=np.float64) * dt
    o2_ref_scale = PHYSICS.typical_o2_total if o2_scale_m3 is None else max(float(o2_scale_m3), 1000.0)

    # 1. \u7eaf\u5355\u5e27\u6ce2\u957f\u7ef4\u7a7a\u95f4\u9664\u6bdb\u523a\uff08\u4ec5\u6cbf\u6ce2\u957f axis=1 \u505a 5 \u70b9\u4e2d\u503c\u6b8b\u5dee\u7b5b\u67e5\uff0c\u96f6\u8de8\u5e27\u4f9d\u8d56\uff01\uff09
    med_w = median_filter(s_raw, size=(1, 5), mode="nearest")
    res_w = s_raw - med_w
    mad_w = np.median(np.abs(res_w), axis=1, keepdims=True) + 15.0
    m_sat = s_raw >= PHYSICS.adc_saturation
    m_spatial_spike = (np.abs(res_w) > 8.0 * mad_w) & (~m_sat)
    s_frame = np.where(m_spatial_spike, med_w, s_raw)

    # 2. \u5355\u5e27\u7d2b\u5916\u6697\u7535\u6d41\u6263\u9664 (< 385 nm) + \u7a84\u7a97\u6ce2\u957f SG \u4fdd\u5cf0\u6ee4\u6ce2 (axis=1)
    dark = np.median(s_frame[:, wavelengths < PHYSICS.dark_wl_max], axis=1, keepdims=True)
    s_net = np.maximum(savgol_filter(s_frame - dark, Q1_CFG.sg_wl_win, Q1_CFG.sg_wl_poly, axis=1), 1.0)

    # 3. \u91cd\u91c7\u6837\u81f3\u516c\u5171\u6ce2\u957f\u7f51\u683c [380, 1010] nm\uff08\u540c\u65f6\u4f20\u64ad\u5355\u5e27\u9971\u548c\u63a9\u819c\uff09
    common_wl = np.linspace(PHYSICS.common_wl_min, PHYSICS.common_wl_max, PHYSICS.common_wl_points)
    n_wl = len(common_wl)
    idx_hi = np.clip(np.searchsorted(wavelengths, common_wl), 1, len(wavelengths) - 1)
    idx_lo = idx_hi - 1
    w_hi = (common_wl - wavelengths[idx_lo]) / (wavelengths[idx_hi] - wavelengths[idx_lo])
    s_grid = s_net[:, idx_lo] * (1.0 - w_hi[None, :]) + s_net[:, idx_hi] * w_hi[None, :]
    sat_grid = m_sat[:, idx_lo] | m_sat[:, idx_hi]

    # 4. \u5355\u5e27 Q1 \u4f4e\u79e9\u5149\u8c31\u57fa\u53f3\u5220\u5931\u6295\u5f71\u6062\u590d + \u7ef4\u6069\u8272\u6e29\u63d0\u53d6\uff08\u4e25\u683c\u4ec5\u4f9d\u8d56\u5f53\u524d\u5e27 k\uff01\uff09
    r_modes = Q1_CFG.q3_modal_rank if q1_basis is None else q1_basis[1].shape[1]
    q1_scores = np.zeros((n, r_modes), dtype=np.float64)
    y_grid_obs = np.log1p(s_grid)
    log_sat_lo = np.log1p(PHYSICS.adc_saturation)
    log_sat_hi = np.log1p(PHYSICS.adc_rec_upper)

    if q1_basis is not None:
        mu_sh, vr = q1_basis  # (127,), (127, r)
        # \u672a\u9971\u548c\u5e27\u53ef\u76f4\u63a5\u7528\u6b63\u4ea4\u57fa\u5feb\u901f\u6295\u5f71\uff1b\u542b\u9971\u548c\u50cf\u7d20\u7684\u5e27\u4ec5\u5728\u672a\u9971\u548c\u5b50\u96c6 \u03a9_k \u4e0a\u505a\u5e26\u5cad\u6b63\u5219\u6700\u5c0f\u4e8c\u4e58\u6295\u5f71\u5e76\u53f3\u5220\u5931\u6062\u590d\u9971\u548c\u533a
        no_sat_rows = ~sat_grid.any(axis=1)
        if no_sat_rows.any():
            yc_ns = y_grid_obs[no_sat_rows] - y_grid_obs[no_sat_rows].mean(axis=1, keepdims=True) - mu_sh[None, :]
            q1_scores[no_sat_rows] = yc_ns @ vr

        if (~no_sat_rows).any():
            reg_eye = np.diag([1e-4] + [0.15] * r_modes)
            for k in np.where(~no_sat_rows)[0]:
                om = ~sat_grid[k]
                if om.sum() >= r_modes + 4:
                    design = np.column_stack([np.ones(om.sum()), vr[om, :]])
                    rhs = y_grid_obs[k, om] - mu_sh[om]
                    theta = np.linalg.solve(design.T @ design + reg_eye, design.T @ rhs)
                    c0, ak = float(theta[0]), theta[1:]
                    q1_scores[k] = ak
                    sk = sat_grid[k]
                    y_hat_sk = c0 + mu_sh[sk] + vr[sk, :] @ ak
                    y_rec_sk = np.clip(np.maximum(y_grid_obs[k, sk], y_hat_sk), log_sat_lo, log_sat_hi)
                    s_grid[k, sk] = np.expm1(y_rec_sk)
                elif k > 0:
                    q1_scores[k] = q1_scores[k - 1]

    cont_mask = (
        ((common_wl >= 480) & (common_wl <= 575))
        | ((common_wl >= 610) & (common_wl <= 750))
        | ((common_wl >= 785) & (common_wl <= 960))
    )
    wl_um = common_wl * 1e-3  # \u7528\u5fae\u7c73\u91cf\u7eb2 O(1) \u907f\u514d 1/\u03bb^2 \u5728 1e12 \u91cf\u7ea7\u653e\u5927\u6761\u4ef6\u6570
    inv_wl_um = 1.0 / wl_um
    y_wien = np.log(s_grid + 1.0) + 5.0 * np.log(wl_um[None, :])

    inv_temp = np.zeros(n)
    wien_resid = np.zeros(n)
    c2_um = PHYSICS.hc_over_kb * 1e6  # 14387.77 \u03bcm\xb7K
    for k in range(n):
        use = cont_mask & (~sat_grid[k])
        if use.sum() < 8:
            use = ~sat_grid[k]
        if use.sum() >= 4:
            x_u = inv_wl_um[use]
            xc = x_u - x_u.mean()
            y_u = y_wien[k, use]
            slp = float((y_u @ xc) / (np.sum(xc ** 2) + 1e-12))
            icp = float(y_u.mean() - slp * x_u.mean())
            # \u9650\u5236\u5728\u7269\u7406\u5408\u7406\u8868\u89c2\u8f90\u5c04\u9006\u6e29\u533a\u95f4 [1/(6000K), 1/(900K)] \u907f\u514d\u6781\u6697\u9996\u5e27\u659c\u7387\u53d1\u6563
            inv_temp[k] = float(np.clip(-slp / c2_um, 1.0e-4, 1.2e-3)) * 1e4
            yhat_u = icp + slp * x_u
            wien_resid[k] = float(np.sqrt(np.mean((y_u - yhat_u) ** 2)))
            if q1_basis is None and sat_grid[k].any():
                sk = sat_grid[k]
                i_wien = np.exp(np.clip(icp + slp * inv_wl_um[sk] - 5.0 * np.log(wl_um[sk]), 0.0, 12.0))
                s_grid[k, sk] = np.clip(np.maximum(s_grid[k, sk], i_wien), PHYSICS.adc_saturation, PHYSICS.adc_rec_upper)
        else:
            inv_temp[k] = inv_temp[k - 1] if k > 0 else 5.0
            wien_resid[k] = wien_resid[k - 1] if k > 0 else 0.1

    # 5. \u76f8\u5bf9\u5149\u8c31\u6982\u7387\u5bc6\u5ea6\u4e0e\u65e0\u91cf\u7eb2\u5316\u8f90\u5c04\u7269\u7406\u7279\u5f81\uff08\u6240\u6709\u5217\u5747\u7f29\u653e\u81f3 O(1)\uff09
    area = s_grid.sum(axis=1, keepdims=True) + 1e-6
    prob = s_grid / area

    wl_norm = (common_wl - 700.0) / 300.0
    wl_mean = prob @ wl_norm
    wl_std = np.sqrt(np.maximum(prob @ (wl_norm ** 2) - wl_mean ** 2, 1e-6))
    spec_ent = -np.sum(prob * np.log(prob + 1e-12), axis=1) / np.log(n_wl)

    def bp(l0: float, l1: float) -> np.ndarray:
        return prob[:, (common_wl >= l0) & (common_wl <= l1)].mean(axis=1) * 100.0

    bands = [
        bp(400, 460), bp(460, 520), bp(520, 580), bp(585, 595), bp(600, 660),
        bp(660, 745), bp(760, 775), bp(785, 860), bp(860, 940), bp(940, 1010),
    ]
    r_na = np.log((bands[3] + 1e-6) / (0.5 * (bands[2] + bands[4]) + 1e-6))
    r_k = np.log((bands[6] + 1e-6) / (0.5 * (bands[5] + bands[7]) + 1e-6))
    r_gr = np.log((bands[2] + 1e-6) / (bands[5] + 1e-6))
    r_rn = np.log((bands[5] + 1e-6) / (bands[7] + 1e-6))
    r_n12 = np.log((bands[7] + 1e-6) / (bands[8] + 1e-6))
    r_br = np.log((bands[1] + 1e-6) / (bands[5] + 1e-6))
    log_area = np.log1p(area[:, 0]) / 10.0
    rel_bright = log_area - np.cumsum(log_area) / (np.arange(n) + 1.0)

    inst = np.column_stack([
        inv_temp, wien_resid, wl_mean, wl_std, spec_ent,
        *bands, r_na, r_k, r_gr, r_rn, r_n12, r_br, rel_bright, log_area,
        q1_scores,
    ])

    # 6. \u4e25\u683c\u56e0\u679c\u524d 60 s \u7089\u6b21\u7269\u7406\u57fa\u7ebf + \u591a\u5c3a\u5ea6\u4e25\u683c\u56e0\u679c EMA
    n60 = max(1, int(round(Q3_CFG.heat_baseline_s / dt)))
    base60 = np.vstack([inst[:min(k + 1, n60)].mean(axis=0) for k in range(n)])
    inst_hn = inst - base60

    emas = [_causal_ema(inst, sc, dt) for sc in Q3_CFG.ema_scales_s]
    hn5, hn25 = _causal_ema(inst_hn, 5.0, dt), _causal_ema(inst_hn, 25.0, dt)
    flicker_idx = [0, 2, 15, 16, 21, 22]
    flicker = _causal_ema(np.abs(inst[:, flicker_idx] - emas[0][:, flicker_idx]), Q3_CFG.flicker_scale_s, dt)

    x_spec = np.hstack([*emas, emas[0] - emas[1], emas[1] - emas[2], flicker])
    x_spec_hn = np.hstack([x_spec, hn5, hn25])

    # 7. \u7eaf\u5355\u4fa7\u56e0\u679c\u7d2f\u79ef\u6c27\u8017\u4e0e\u3010\u53bb PLC \u8865\u8d26\u8df3\u53d8\u3011\u7684\u56e0\u679c 24 s \u6c27\u67aa\u77ac\u65f6\u6d41\u91cf [P1-4 & P0-5 \u4fee\u590d]
    o2_mon = np.maximum.accumulate(np.maximum(o2_raw.astype(np.float64), 0.0))
    o2_causal = np.maximum.accumulate(_causal_ema(o2_mon[:, None], 3.0, dt)[:, 0])

    # \u5bf9\u6d41\u91cf\u4f30\u8ba1\u65bd\u52a0\u5355\u6b65\u56e0\u679c\u8df3\u53d8\u8fc7\u6ee4\uff08\u4ec5\u66ff\u6362 > 14.5 m\xb3 \u7684\u65ad\u884c\u8865\u8d26\u5927\u8df3\u53d8\uff0c\u5b8c\u6574\u4fdd\u7559 <= 10.5 m\xb3 \u7684\u6b63\u5e38 1s/\u62cd\u9891\u8109\u51b2\uff0c100% \u56e0\u679c\uff01\uff09
    max_step = PHYSICS.o2_gap_jump_m3
    do2_c = np.diff(o2_mon, prepend=o2_mon[0])
    do2_flow_c = np.empty(n)
    running_mean_step = 2.35 * dt
    for k in range(n):
        if do2_c[k] > max_step:
            do2_flow_c[k] = running_mean_step
        else:
            do2_flow_c[k] = do2_c[k]
            running_mean_step = 0.94 * running_mean_step + 0.06 * do2_c[k]
    o2_flow_cum = o2_mon[0] + np.cumsum(do2_flow_c)

    w24 = max(2, int(round(24.0 / dt)))
    q_causal = np.empty(n)
    q_causal[:w24] = (o2_flow_cum[:w24] - o2_flow_cum[0]) / np.maximum(t[:w24], dt)
    q_causal[0] = q_causal[1] if n > 1 else 2.35
    q_causal[w24:] = (o2_flow_cum[w24:] - o2_flow_cum[:-w24]) / (w24 * dt)
    q_causal = np.clip(_causal_ema(q_causal[:, None], 3.0, dt)[:, 0], 0.0, PHYSICS.max_phys_o2_rate)

    o2_n = np.clip(o2_causal / o2_ref_scale, 0.0, 1.35)
    q_sm30 = _causal_ema(q_causal[:, None], 30.0, dt)[:, 0] / 5.0
    # \u6709\u754c\u5206\u6bb5\u94f0\u94fe\u57fa\u66ff\u4ee3\u53d1\u6563\u4e09\u6b21\u591a\u9879\u5f0f\uff0c\u517c\u987e\u9636\u6bb5\u975e\u7ebf\u6027\u4e0e\u9ad8\u6c27\u8017\u5916\u63a8\u7a33\u5b9a\u6027
    hinges = np.column_stack([
        np.clip(o2_n - 0.18, 0.0, 1.2),
        np.clip(o2_n - 0.42, 0.0, 1.0),
        np.clip(o2_n - 0.85, 0.0, 0.6),
    ])
    x_proc = np.column_stack([
        o2_n, np.log1p(o2_causal / 200.0), q_causal / 5.0, q_sm30, hinges,
    ])

    # \u4f9b\u6709\u754c\u7ebf\u6027\u8d8b\u52bf\u5916\u63a8\u5934\u4f7f\u7528\u7684\u5e73\u6ed1\u7279\u5f81\u5b50\u96c6
    x_lin = np.hstack([emas[1], hn25, x_proc])

    # \u4e25\u683c\u4ec5\u7531\u524d 60 s \u56e0\u679c\u89c2\u6d4b\u7ec4\u6210\u7684\u7089\u6b21\u5de5\u51b5\u6307\u7eb9\uff08\u96f6\u672a\u6765\u5e27\u6cc4\u6f0f\uff01\uff09
    k60 = min(n, n60)
    desc_c60 = np.r_[
        prob[:k60].mean(axis=0)[::8] * 100.0,
        q1_scores[:k60].mean(axis=0),
        float(q_causal[:k60].mean()),
        float(log_area[:k60].mean()),
    ]

    # 8. \u5728\u7ebf\u5206\u6790\u4eea\u6ede\u540e\u8bfb\u6570\u7684\u5355\u4fa7\u56e0\u679c\u53bb\u566a\u4e0e\u539f\u59cb ZOH \u5e8f\u5217
    raw_co2 = np.clip(co2_raw.astype(np.float64), 0.0, None)
    raw_co = np.clip(co_raw.astype(np.float64), 0.0, None)
    obs_co2_c = _causal_ema(raw_co2[:, None], 2.5, dt)[:, 0]
    obs_co_c = _causal_ema(raw_co[:, None], 2.5, dt)[:, 0]

    return CausalOnlineStreams(
        heat_id=heat_id, heat_no=heat_no, converter=heat_no[:2] if heat_no else "", dt=dt, time_s=t,
        spec_prob=prob, q1_modal_scores=q1_scores, x_proc=x_proc, x_spec=x_spec,
        x_m3=np.hstack([x_spec, x_proc]), x_m4=np.hstack([x_spec_hn, x_proc]),
        x_lin=x_lin, desc_causal60=desc_c60,
        o2_causal=o2_causal, q_causal=q_causal,
        obs_co2_causal=obs_co2_c, obs_co_causal=obs_co_c,
        raw_co2_zoh=raw_co2, raw_co_zoh=raw_co,
    )


def build_targets_and_backward_map(
    stream: CausalOnlineStreams,
    q2_recon: Q2HeatRecon,
    delay: PFRDelayParams,
) -> Dict[str, np.ndarray]:
    """
    \u6839\u636e\u8bad\u7ec3\u6298\u72ec\u7acb\u6807\u5b9a\u7684\u5e73\u63a8\u6d41\u53c2\u6570 delay = \u0394V_pipe^{(-i)}\uff1a
      - \u79bb\u7ebf\u771f\u5b9e\u5439\u70bc\u72b6\u6001\u6807\u7b7e\uff08\u4ec5\u7528\u4e8e\u76d1\u7763\u8bad\u7ec3\u4e0e\u79bb\u7ebf\u7cbe\u5ea6\u8bc4\u4f30\uff09\uff1ay_al(t) = y_clean(t_arr(t))
      - \u5728\u7ebf\u4e25\u683c\u56e0\u679c\u53cd\u5411\u67e5\u8be2\u7d22\u5f15 k_src[k] <= k\uff08\u4ec5\u67e5\u8be2\u56e0\u679c\u524d\u7f00 o2_causal[:k+1]\uff01\uff09
    """
    t, dt = stream.time_s, stream.dt
    n = len(t)
    tau_base = delay.tau0_s
    v_pipe = delay.v_pipe_m3

    o2_off = q2_recon.o2_cont + np.arange(n) * 1e-6
    t_arr = np.interp(q2_recon.o2_cont + v_pipe, o2_off, t, right=np.nan) + tau_base
    valid = (~np.isnan(t_arr)) & (t_arr <= t[-1])

    y_co2 = np.full(n, np.nan)
    y_co = np.full(n, np.nan)
    y_co2[valid] = np.interp(t_arr[valid], t, q2_recon.co2_cont)
    y_co[valid] = np.interp(t_arr[valid], t, q2_recon.co_cont)

    o2_c_u = stream.o2_causal + np.arange(n) * 1e-6
    k_src = np.full(n, -1, dtype=int)
    for k in range(n):
        t1 = t[k] - tau_base
        if t1 < 0:
            continue
        o2_t1 = float(np.interp(t1, t[:k + 1], o2_c_u[:k + 1]))
        target_o2 = o2_t1 - v_pipe
        if target_o2 >= o2_c_u[0]:
            ts = float(np.interp(target_o2, o2_c_u[:k + 1], t[:k + 1]))
            k_src[k] = min(k, int(round(ts / dt)))

    return {
        "t_arr": t_arr,
        "valid": valid,
        "y_co2": y_co2,
        "y_co": y_co,
        "k_src": k_src,
    }


def _fit_predict_linear_tree(
    x_tree_tr: np.ndarray,
    x_lin_tr: np.ndarray,
    y1_tr: np.ndarray,
    y2_tr: np.ndarray,
    x_tree_te: np.ndarray,
    x_lin_te: np.ndarray,
    blend1: float,
    blend2: float,
    fast_mode: bool = False,
    sample_weight: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    \u7ebf\u6027\u8d8b\u52bf\u5916\u63a8\u5934 + \u591a\u6df1\u5ea6 XGBoost \u6b8b\u5dee\u6811\u6df7\u5408\u6a21\u578b\uff08Linear-Tree Hybrid\uff0c\u652f\u6301\u56e0\u679c\u76f8\u4f3c\u5ea6\u6837\u672c\u52a0\u6743\uff09\uff1a
    \u514b\u670d\u7eaf\u51b3\u7b56\u6811\u9636\u68af\u5e38\u6570\u8f93\u51fa\u5728\u5206\u5e03\u5916\u9ad8\u78b3\u5cf0\u503c\uff08\u5982\u65b0\u7248 H01 \u7684 59% CO \u5cf0\u503c\uff09\u5904\u65e0\u6cd5\u5411\u4e0a\u5916\u63a8\u7684\u56fa\u6709\u77ed\u677f\uff0c
    \u540c\u65f6\u675c\u7edd\u65e7\u7248\u62c9\u683c\u6717\u65e5\u6295\u5f71\u5c06\u9ad8\u566a CO2 \u8bef\u5dee\u53cd\u5411\u6c61\u67d3 CO \u7684\u526f\u4f5c\u7528\u3002
    """
    if fast_mode:
        # \u5185\u5c42 8-vs-1 OOF \u6807\u5b9a\u4f7f\u7528 2 \u500d\u5747\u5300\u6b65\u8fdb\u4e0e\u5355\u6df1\u5ea6\u76f4\u65b9\u56fe\u6811\uff0c\u517c\u987e\u9ad8\u4fdd\u771f\u4e0e 5x \u52a0\u901f
        x_tree_tr = x_tree_tr[::2]
        x_lin_tr = x_lin_tr[::2]
        y1_tr = y1_tr[::2]
        y2_tr = y2_tr[::2]
        if sample_weight is not None:
            sample_weight = sample_weight[::2]

    depths = (4,) if fast_mode else Q3_CFG.xgb_depths
    seeds = (SEED,) if fast_mode else Q3_CFG.xgb_seeds
    n_est = 95 if fast_mode else Q3_CFG.xgb_n_estimators

    # \u7eaf\u591a\u6df1\u5ea6\u6811\u96c6\u6210\u9884\u6d4b\uff08\u4f7f\u7528 hist \u76f4\u65b9\u56fe\u52a0\u901f\uff09
    p1_pure_list, p2_pure_list = [], []
    for depth, seed in zip(depths, seeds):
        kw = dict(
            n_estimators=n_est, max_depth=depth, learning_rate=Q3_CFG.xgb_lr,
            subsample=0.8, colsample_bytree=0.65, reg_lambda=Q3_CFG.xgb_reg_lambda,
            tree_method="hist", random_state=seed, n_jobs=4,
        )
        p1_pure_list.append(xgb.XGBRegressor(**kw).fit(x_tree_tr, y1_tr, sample_weight=sample_weight).predict(x_tree_te))
        p2_pure_list.append(xgb.XGBRegressor(**kw).fit(x_tree_tr, y2_tr, sample_weight=sample_weight).predict(x_tree_te))
    p1_pure = np.mean(p1_pure_list, axis=0)
    p2_pure = np.mean(p2_pure_list, axis=0)

    if blend1 <= 0.0 and blend2 <= 0.0:
        return np.clip(p1_pure, 0.0, None), np.clip(p2_pure, 0.0, None)

    # \u6709\u754c tanh \u8f6f\u9650\u5e45\u7ebf\u6027\u8d8b\u52bf\u5934 + \u6811\u62df\u5408\u975e\u7ebf\u6027\u6b8b\u5dee
    sc = StandardScaler().fit(x_lin_tr)
    xl_tr = 3.0 * np.tanh(sc.transform(x_lin_tr) / 3.0)
    xl_te = 3.0 * np.tanh(sc.transform(x_lin_te) / 3.0)

    kw_res = dict(
        n_estimators=n_est, max_depth=4, learning_rate=Q3_CFG.xgb_lr,
        subsample=0.8, colsample_bytree=0.65, reg_lambda=Q3_CFG.xgb_reg_lambda,
        tree_method="hist", random_state=SEED, n_jobs=4,
    )
    if blend1 > 0.0:
        lin1 = Ridge(alpha=Q3_CFG.ridge_alpha).fit(xl_tr, y1_tr, sample_weight=sample_weight)
        res1_hat = xgb.XGBRegressor(**kw_res).fit(
            x_tree_tr, y1_tr - lin1.predict(xl_tr), sample_weight=sample_weight
        ).predict(x_tree_te)
        p1_hyb = lin1.predict(xl_te) + res1_hat
        p1_out = (1.0 - blend1) * p1_pure + blend1 * p1_hyb
    else:
        p1_out = p1_pure

    if blend2 > 0.0:
        lin2 = Ridge(alpha=Q3_CFG.ridge_alpha).fit(xl_tr, y2_tr, sample_weight=sample_weight)
        res2_hat = xgb.XGBRegressor(**kw_res).fit(
            x_tree_tr, y2_tr - lin2.predict(xl_tr), sample_weight=sample_weight
        ).predict(x_tree_te)
        p2_hyb = lin2.predict(xl_te) + res2_hat
        p2_out = (1.0 - blend2) * p2_pure + blend2 * p2_hyb
    else:
        p2_out = p2_pure

    return np.clip(p1_out, 0.0, None), np.clip(p2_out, 0.0, None)


def _compute_causal_heat_weights(
    streams: List[CausalOnlineStreams],
    train_idx: List[int],
    test_idx: int,
) -> np.ndarray:
    """
    \u57fa\u4e8e\u5f00\u5439\u524d 60 s \u4e25\u683c\u56e0\u679c\u5de5\u51b5\u6307\u7eb9 + \u8f6c\u7089\u7089\u578b\u4eb2\u548c\u5ea6\u8ba1\u7b97\u5404\u8bad\u7ec3\u7089\u7684\u5e73\u6ed1\u6838\u6743\u91cd\uff1a
    \u65e2\u4fdd\u7559\u5168\u90e8\u8bad\u7ec3\u7089\u7684\u6837\u672c\u91cf\u4ee5\u6291\u5236\u5355\u7089\u65b9\u5dee\uff0c\u53c8\u5bf9\u5de5\u51b5\u76f8\u8fd1\u7684\u540c\u7fa4\u7089\u6b21\uff08\u5982 H01 \u4e0e H03\uff09\u7ed9\u4e88\u66f4\u9ad8\u62df\u5408\u6743\u91cd\u3002
    """
    d_tr = np.vstack([streams[j].desc_causal60 for j in train_idx])
    sc_d = StandardScaler().fit(d_tr)
    z_tr = sc_d.transform(d_tr)
    z_te = sc_d.transform(streams[test_idx].desc_causal60[None, :])[0]

    sq_dists = np.sum((z_tr - z_te[None, :]) ** 2, axis=1)
    same_conv = np.array([1.0 if streams[j].converter == streams[test_idx].converter else 0.0 for j in train_idx])
    scale_d = max(float(np.median(sq_dists)), 1.0)
    logits = -0.65 * (sq_dists / scale_d) + 0.55 * same_conv
    logits -= logits.max()
    soft_w = np.exp(logits)
    soft_w /= soft_w.sum()
    # \u6df7\u5408\u5747\u5300\u5148\u9a8c (0.55) \u4e0e\u56e0\u679c\u6838\u4eb2\u548c\u5206\u5e03 (0.45)\uff0c\u786e\u4fdd\u5e73\u6ed1\u9632\u8fc7\u62df\u5408\u4e14\u6709\u6548\u6837\u672c\u6570\u5145\u8db3
    return 0.55 + 0.45 * (len(train_idx) * soft_w)


def apply_causal_smith_observer(
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    obs_co2_c: np.ndarray,
    obs_co_c: np.ndarray,
    k_src: np.ndarray,
    dt: float,
    interval_s: float | None,
    gain: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """100% \u4e25\u683c\u5355\u4fa7\u56e0\u679c Smith \u4e92\u8865\u89c2\u6d4b\u5668\u3002"""
    if interval_s is None or gain <= 0.0:
        return p1_b.copy(), p2_b.copy()
    n = len(p1_b)
    period = max(1, int(round(interval_s / dt)))
    alpha_b = (1.0 - np.exp(-dt / Q3_CFG.smith_tau_s)) if period == 1 else 0.85
    b1, b2 = np.zeros(n), np.zeros(n)
    for k in range(n):
        ks = k_src[k]
        if ks >= 0 and (k % period == 0):
            e1 = obs_co2_c[k] - p1_b[ks]
            e2 = obs_co_c[k] - p2_b[ks]
            b1[k] = alpha_b * e1 + (1.0 - alpha_b) * (b1[k - 1] if k > 0 else e1)
            b2[k] = alpha_b * e2 + (1.0 - alpha_b) * (b2[k - 1] if k > 0 else e2)
        else:
            b1[k] = b1[k - 1] if k > 0 else 0.0
            b2[k] = b2[k - 1] if k > 0 else 0.0
    return np.clip(p1_b + gain * b1, 0.0, None), np.clip(p2_b + gain * b2, 0.0, None)


def apply_naive_lagged_sensor(
    raw_co2_zoh: np.ndarray,
    raw_co_zoh: np.ndarray,
    dt: float,
    interval_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [P1-1 \u65b0\u589e\u57fa\u7ebf] \u672a\u7ecf\u53d8\u65f6\u6ede\u5bf9\u9f50\u4e0e\u5149\u8c31\u8f6f\u6d4b\u91cf\u878d\u5408\u7684\u7eaf\u786c\u4ef6\u4f20\u611f\u5668\u96f6\u9636\u4fdd\u6301\u6301\u4e45\u6027\u57fa\u7ebf\uff1a
    \u6bcf\u9694 interval_s \u79d2\u8bfb\u53d6\u4e00\u6b21\u70df\u9053\u5206\u6790\u4eea\u5f53\u524d\u503c\u5e76\u4fdd\u6301\u81f3\u4e0b\u4e00\u6b21\u91c7\u6837\u3002
    """
    n = len(raw_co2_zoh)
    period = max(1, int(round(interval_s / dt)))
    p1, p2 = np.empty(n), np.empty(n)
    c1, c2 = raw_co2_zoh[0], raw_co_zoh[0]
    for k in range(n):
        if k % period == 0:
            c1 = raw_co2_zoh[k]
            c2 = raw_co_zoh[k]
        p1[k] = c1
        p2[k] = c2
    return p1, p2


def _eval_fold_model_variant(
    streams_fold: List[CausalOnlineStreams],
    q2_recons: List[Q2HeatRecon],
    maps_fold: List[Dict[str, np.ndarray]],
    train_idx: List[int],
    test_idx: int,
    variant: str,
    fast_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """\u5728\u4e25\u683c\u9694\u79bb\u6d4b\u8bd5\u7089\u7684\u5355\u4e00\u6298\u7279\u5f81\u6d41\u4e0a\u8bad\u7ec3\u5e76\u9884\u6d4b\u6307\u5b9a\u6d88\u878d/\u7ec4\u5408\u53d8\u4f53\u3002"""
    if variant == "M0_ProcessOnly":
        xtr = np.vstack([streams_fold[j].x_proc[maps_fold[j]["valid"]] for j in train_idx])
        y1 = np.concatenate([maps_fold[j]["y_co2"][maps_fold[j]["valid"]] for j in train_idx])
        y2 = np.concatenate([maps_fold[j]["y_co"][maps_fold[j]["valid"]] for j in train_idx])
        return _fit_predict_linear_tree(xtr, xtr, y1, y2, streams_fold[test_idx].x_proc, streams_fold[test_idx].x_proc, 0.0, 0.0, fast_mode)

    if variant == "M1_SpectraAndQ1Modes":
        xtr = np.vstack([streams_fold[j].x_spec[maps_fold[j]["valid"]] for j in train_idx])
        y1 = np.concatenate([maps_fold[j]["y_co2"][maps_fold[j]["valid"]] for j in train_idx])
        y2 = np.concatenate([maps_fold[j]["y_co"][maps_fold[j]["valid"]] for j in train_idx])
        return _fit_predict_linear_tree(xtr, xtr, y1, y2, streams_fold[test_idx].x_spec, streams_fold[test_idx].x_spec, 0.0, 0.0, fast_mode)

    if variant == "M2_SpecProc_Unaligned":
        xtr = np.vstack([streams_fold[j].x_m3[maps_fold[j]["valid"]] for j in train_idx])
        y1 = np.concatenate([q2_recons[j].co2_cont[maps_fold[j]["valid"]] for j in train_idx])
        y2 = np.concatenate([q2_recons[j].co_cont[maps_fold[j]["valid"]] for j in train_idx])
        return _fit_predict_linear_tree(xtr, xtr, y1, y2, streams_fold[test_idx].x_m3, streams_fold[test_idx].x_m3, 0.0, 0.0, fast_mode)

    if variant == "M3_SpecProc_Aligned":
        xtr = np.vstack([streams_fold[j].x_m3[maps_fold[j]["valid"]] for j in train_idx])
        y1 = np.concatenate([maps_fold[j]["y_co2"][maps_fold[j]["valid"]] for j in train_idx])
        y2 = np.concatenate([maps_fold[j]["y_co"][maps_fold[j]["valid"]] for j in train_idx])
        return _fit_predict_linear_tree(
            xtr, streams_fold[test_idx].x_m3[:0] if False else np.vstack([streams_fold[j].x_lin[maps_fold[j]["valid"]] for j in train_idx]),
            y1, y2, streams_fold[test_idx].x_m3, streams_fold[test_idx].x_lin,
            Q3_CFG.linear_blend_co2, Q3_CFG.linear_blend_co, fast_mode,
        )

    if variant == "M4_Whole_LinearTree":
        xtr = np.vstack([streams_fold[j].x_m4[maps_fold[j]["valid"]] for j in train_idx])
        xl_tr = np.vstack([streams_fold[j].x_lin[maps_fold[j]["valid"]] for j in train_idx])
        y1 = np.concatenate([maps_fold[j]["y_co2"][maps_fold[j]["valid"]] for j in train_idx])
        y2 = np.concatenate([maps_fold[j]["y_co"][maps_fold[j]["valid"]] for j in train_idx])
        return _fit_predict_linear_tree(
            xtr, xl_tr, y1, y2, streams_fold[test_idx].x_m4, streams_fold[test_idx].x_lin,
            Q3_CFG.linear_blend_co2, Q3_CFG.linear_blend_co, fast_mode,
        )

    if variant == "Grouped_Converter_6B6C":
        sub_idx = [j for j in train_idx if streams_fold[j].converter == streams_fold[test_idx].converter]
        xtr = np.vstack([streams_fold[j].x_m4[maps_fold[j]["valid"]] for j in sub_idx])
        xl_tr = np.vstack([streams_fold[j].x_lin[maps_fold[j]["valid"]] for j in sub_idx])
        y1 = np.concatenate([maps_fold[j]["y_co2"][maps_fold[j]["valid"]] for j in sub_idx])
        y2 = np.concatenate([maps_fold[j]["y_co"][maps_fold[j]["valid"]] for j in sub_idx])
        return _fit_predict_linear_tree(
            xtr, xl_tr, y1, y2, streams_fold[test_idx].x_m4, streams_fold[test_idx].x_lin,
            Q3_CFG.linear_blend_co2, Q3_CFG.linear_blend_co, fast_mode,
        )

    if variant == "Grouped_CausalWardCluster":
        # \u4e25\u683c\u4ec5\u5728 train_idx \u7684\u524d 60 s \u56e0\u679c\u6307\u7eb9\u4e0a\u505a Ward 2-\u805a\u7c7b\uff0c\u6d4b\u8bd5\u7089\u6309\u6700\u8fd1\u56e0\u679c\u7c07\u8d28\u5fc3\u5f52\u7ec4\uff01
        d_tr = np.vstack([streams_fold[j].desc_causal60 for j in train_idx])
        sc_c = StandardScaler().fit(d_tr)
        z_tr = sc_c.transform(d_tr)
        cl_labels = fcluster(linkage(z_tr, method="ward"), t=2, criterion="maxclust")
        z_te = sc_c.transform(streams_fold[test_idx].desc_causal60[None, :])[0]
        c_dists = [
            float(np.linalg.norm(z_te - z_tr[cl_labels == cid].mean(axis=0))) if np.any(cl_labels == cid) else np.inf
            for cid in (1, 2)
        ]
        best_cid = 1 if c_dists[0] <= c_dists[1] else 2
        sub_idx = [train_idx[pos] for pos in range(len(train_idx)) if cl_labels[pos] == best_cid]
        if len(sub_idx) < 2:
            sub_idx = train_idx
        xtr = np.vstack([streams_fold[j].x_m4[maps_fold[j]["valid"]] for j in sub_idx])
        xl_tr = np.vstack([streams_fold[j].x_lin[maps_fold[j]["valid"]] for j in sub_idx])
        y1 = np.concatenate([maps_fold[j]["y_co2"][maps_fold[j]["valid"]] for j in sub_idx])
        y2 = np.concatenate([maps_fold[j]["y_co"][maps_fold[j]["valid"]] for j in sub_idx])
        return _fit_predict_linear_tree(
            xtr, xl_tr, y1, y2, streams_fold[test_idx].x_m4, streams_fold[test_idx].x_lin,
            Q3_CFG.linear_blend_co2, Q3_CFG.linear_blend_co, fast_mode,
        )

    if variant == "Combined_SoftGated_Hybrid":
        # \u524d 60 s \u56e0\u679c\u6838\u52a0\u6743\u5168\u7089\u7ebf\u6027\u6811\u6df7\u5408\uff08Causal Kernel-Weighted Pooled Hybrid\uff09\uff1a
        # \u65e2\u907f\u514d\u786c\u5206\u7ec4/\u5355\u7089\u4e13\u5bb6\u7684\u5c0f\u6837\u672c\u65b9\u5dee\u7206\u70b8\uff0c\u53c8\u8f6f\u6027\u5f3a\u5316\u76f8\u4f3c\u5de5\u51b5\u4e0e\u540c\u7089\u578b\u8bad\u7ec3\u6837\u672c\u7684\u8d21\u732e\uff01
        heat_wts = _compute_causal_heat_weights(streams_fold, train_idx, test_idx)
        xtr = np.vstack([streams_fold[j].x_m4[maps_fold[j]["valid"]] for j in train_idx])
        xl_tr = np.vstack([streams_fold[j].x_lin[maps_fold[j]["valid"]] for j in train_idx])
        y1 = np.concatenate([maps_fold[j]["y_co2"][maps_fold[j]["valid"]] for j in train_idx])
        y2 = np.concatenate([maps_fold[j]["y_co"][maps_fold[j]["valid"]] for j in train_idx])
        sw = np.concatenate([np.full(int(maps_fold[j]["valid"].sum()), w_j) for j, w_j in zip(train_idx, heat_wts)])
        return _fit_predict_linear_tree(
            xtr, xl_tr, y1, y2, streams_fold[test_idx].x_m4, streams_fold[test_idx].x_lin,
            Q3_CFG.linear_blend_co2, Q3_CFG.linear_blend_co, fast_mode, sample_weight=sw,
        )

    raise ValueError(f"Unknown model variant: {variant}")


def run_question3(
    heats: List[HeatDataset],
    q1_res: List[Q1Result],
    q2_sum: Q2Summary,
    make_plots: bool = True,
) -> Dict[str, pd.DataFrame]:
    q2_recons = q2_sum.recons

    # 1. \u5916\u5c42 10 \u6298\u7559\u4e00\u7089\uff08LOHO-CV\uff09\uff1a\u6bcf\u6298\u72ec\u7acb\u4ece 9 \u4e2a\u8bad\u7ec3\u7089\u63d0\u53d6 (a) Q1 \u5149\u8c31\u57fa (\u03bc^{(-i)}, V_r^{(-i)})\uff0c(b) PFR \u65f6\u5ef6 \u0394V_pipe^{(-i)}\uff0c(c) \u8017\u6c27\u6807\u5ea6\uff01
    fold_delays: List[PFRDelayParams] = []
    fold_bases: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_o2_scales: List[float] = []
    fold_delay_rows = []

    for i in range(10):
        tr_idx = [j for j in range(10) if j != i]
        fd = fit_pfr_kinetics(q2_recons, tr_idx, f"Fold_Excl_H{i+1:02d}", run_bootstrap=False)
        basis_i = extract_fold_q1_spectral_basis(q1_res, heats, tr_idx, n_modes=Q1_CFG.q3_modal_rank)
        o2_sc_i = float(np.mean([q2_recons[j].o2_cont[-1] for j in tr_idx]))
        fold_delays.append(fd)
        fold_bases.append(basis_i)
        fold_o2_scales.append(o2_sc_i)
        fold_delay_rows.append({
            "Test_Heat": f"H{i+1:02d}",
            "Train_O2_Scale_m3": round(o2_sc_i, 2),
            "Train_dV_pipe_m3": round(fd.v_pipe_m3, 2),
            "Train_Tmix_s": fd.t_mix_s,
            "Train_PFR_R2": round(fd.mean_r2, 4),
        })
    pd.DataFrame(fold_delay_rows).to_csv(TABLE_DIR / "q3_per_fold_delay_params.csv", index=False)

    # 2. \u4e3a\u6bcf\u4e00\u6298\u6784\u9020\u4e25\u683c\u57fa\u4e8e\u8be5\u6298\u8bad\u7ec3\u4fa7 (basis_i, o2_sc_i, fd_i) \u7684\u56e0\u679c\u7279\u5f81\u6d41\u4e0e\u53cd\u5411\u6620\u5c04\uff08\u96f6\u8de8\u7089\u6e17\u900f\uff01\uff09
    per_fold_streams: List[List[CausalOnlineStreams]] = []
    per_fold_maps: List[List[Dict[str, np.ndarray]]] = []
    for i in range(10):
        st_i = [
            compute_strictly_causal_streams(
                h.spectrum, h.wavelengths, h.o2_cum, h.co2_raw, h.co_raw, h.dt,
                h.heat_id, h.heat_no, q1_basis=fold_bases[i], o2_scale_m3=fold_o2_scales[i],
            )
            for h in heats
        ]
        mp_i = [build_targets_and_backward_map(st_i[j], q2_recons[j], fold_delays[i]) for j in range(10)]
        per_fold_streams.append(st_i)
        per_fold_maps.append(mp_i)

    # \u6bcf\u4e2a\u6d4b\u8bd5\u7089\u5bf9\u5e94\u7684\u65e0\u6cc4\u6f0f\u6d4b\u8bd5\u6d41\u4e0e\u76d1\u7763\u8bc4\u4f30\u6620\u5c04
    test_streams = [per_fold_streams[i][i] for i in range(10)]
    test_maps = [per_fold_maps[i][i] for i in range(10)]

    # 3. 10\xd710 \u5355\u7089\u8fc1\u79fb\u77e9\u9635\uff08\u8bad\u7ec3\u7089 i \u5355\u7089\u63d0\u53d6\u5355\u7089\u5149\u8c31\u57fa\u4e0e\u5355\u7089\u6a21\u578b \u2192 \u8de8\u7089\u9884\u6d4b j\uff0c\u4f7f\u7528 hist \u52a0\u901f\uff09
    transfer_mat = np.zeros((10, 10))
    kw_s = dict(n_estimators=100, max_depth=3, learning_rate=0.055, subsample=0.8,
                colsample_bytree=0.7, reg_lambda=15.0, tree_method="hist", random_state=SEED, n_jobs=4)
    for i in range(10):
        basis_single = extract_fold_q1_spectral_basis(q1_res, heats, [i], n_modes=Q1_CFG.q3_modal_rank)
        o2_single = float(q2_recons[i].o2_cont[-1])
        s_tr_i = compute_strictly_causal_streams(
            heats[i].spectrum, heats[i].wavelengths, heats[i].o2_cum, heats[i].co2_raw, heats[i].co_raw,
            heats[i].dt, heats[i].heat_id, heats[i].heat_no, q1_basis=basis_single, o2_scale_m3=o2_single,
        )
        for j in range(10):
            delay_use = q2_sum.pfr_global if j == i else fold_delays[j]
            m_tr_i = build_targets_and_backward_map(s_tr_i, q2_recons[i], delay_use)
            vi = m_tr_i["valid"]
            m1 = xgb.XGBRegressor(**kw_s).fit(s_tr_i.x_m4[vi], m_tr_i["y_co2"][vi])
            m2 = xgb.XGBRegressor(**kw_s).fit(s_tr_i.x_m4[vi], m_tr_i["y_co"][vi])
            s_te_j = compute_strictly_causal_streams(
                heats[j].spectrum, heats[j].wavelengths, heats[j].o2_cum, heats[j].co2_raw, heats[j].co_raw,
                heats[j].dt, heats[j].heat_id, heats[j].heat_no, q1_basis=basis_single, o2_scale_m3=o2_single,
            )
            m_te_j = build_targets_and_backward_map(s_te_j, q2_recons[j], delay_use)
            vj = m_te_j["valid"]
            p1 = np.clip(m1.predict(s_te_j.x_m4[vj]), 0.0, None)
            p2 = np.clip(m2.predict(s_te_j.x_m4[vj]), 0.0, None)
            transfer_mat[i, j] = 0.5 * (r2_score(m_te_j["y_co2"][vj], p1) + r2_score(m_te_j["y_co"][vj], p2))

    labels = [f"H{h.heat_id:02d}({h.heat_no})" for h in heats]
    df_transfer = pd.DataFrame(transfer_mat, index=labels, columns=labels)
    df_transfer.to_csv(TABLE_DIR / "q3_single_heat_transfer_matrix.csv")

    # 4. \u5168\u6d88\u878d\u4e0e\u56db\u79cd\u591a\u7089\u7ec4\u5408\u67b6\u6784\u5bf9\u6bd4\uff08\u4e25\u683c\u5916\u5c42 LOHO\uff09
    variants = [
        "M0_ProcessOnly",
        "M1_SpectraAndQ1Modes",
        "M2_SpecProc_Unaligned",
        "M3_SpecProc_Aligned",
        "M4_Whole_LinearTree",
        "Grouped_Converter_6B6C",
        "Grouped_CausalWardCluster",
        "Combined_SoftGated_Hybrid",
    ]
    preds_map: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {v: [] for v in variants}

    for i in range(10):
        tr_idx = [j for j in range(10) if j != i]
        for vname in variants:
            preds_map[vname].append(
                _eval_fold_model_variant(per_fold_streams[i], q2_recons, per_fold_maps[i], tr_idx, i, vname, fast_mode=False)
            )

    # 5. [P0-6 \u6838\u5fc3\u4fee\u590d] \u4e25\u683c\u5185\u5c42 8-vs-1 \u5d4c\u5957\u4ea4\u53c9\u9a8c\u8bc1\uff08Inner-OOF\uff09\uff1a
    #    \u5728\u5916\u5c42\u6298 i \u7684 9 \u4e2a\u8bad\u7ec3\u7089 tr_idx \u5185\u72ec\u7acb\u8ba1\u7b97\u4e0d\u89e6\u78b0\u6d4b\u8bd5\u7089 i \u7684 Inner-OOF \u9884\u6d4b\uff0c
    #    \u4e13\u95e8\u7528\u4e8e\u65e0\u504f\u6807\u5b9a Smith \u4e92\u8865\u89c2\u6d4b\u5668\u589e\u76ca g^{(-i)} \u4e0e\u5c40\u90e8\u81ea\u9002\u5e94\u5171\u5f62\u9884\u6d4b\u5206\u4f4d\u6570 q_{0.90}^{(-i)}\uff01
    main_zero_preds = preds_map["Combined_SoftGated_Hybrid"]
    inner_oof_zero: List[Dict[int, Tuple[np.ndarray, np.ndarray]]] = []
    for i in range(10):
        tr_idx = [j for j in range(10) if j != i]
        oof_dict: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for j in tr_idx:
            in_tr = [k for k in tr_idx if k != j]
            oof_dict[j] = _eval_fold_model_variant(
                per_fold_streams[i], q2_recons, per_fold_maps[i], in_tr, j, "Combined_SoftGated_Hybrid", fast_mode=True
            )
        inner_oof_zero.append(oof_dict)

    # \u57fa\u4e8e Inner-OOF \u65e0\u6cc4\u6f0f\u6807\u5b9a\u6bcf\u4e2a\u91c7\u6837\u5468\u671f\u7684 Smith \u589e\u76ca\uff0c\u5e76\u540c\u6b65\u8ba1\u7b97 Naive_Lagged_Sensor \u786c\u4ef6\u76f4\u8bfb\u57fa\u7ebf [P1-1]
    smith_preds: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    naive_preds: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    inner_oof_dual: List[Dict[int, Tuple[np.ndarray, np.ndarray]]] = [{} for _ in range(10)]

    for m_name, interval_s in Q3_CFG.probe_intervals_s:
        if interval_s is None:
            smith_preds[m_name] = main_zero_preds
            continue
        per_heat_smith = []
        per_heat_naive = []
        for i in range(10):
            tr_idx = [j for j in range(10) if j != i]
            best_g, best_loss = 0.85, np.inf
            for g_cand in Q3_CFG.smith_gain_candidates:
                losses = []
                for j in tr_idx:
                    sj = per_fold_streams[i][j]
                    mj = per_fold_maps[i][j]
                    vj = mj["valid"]
                    p1_oof, p2_oof = inner_oof_zero[i][j]
                    q1j, q2j = apply_causal_smith_observer(
                        p1_oof, p2_oof, sj.obs_co2_causal, sj.obs_co_causal, mj["k_src"], sj.dt, interval_s, g_cand
                    )
                    losses.append(
                        np.sqrt(mean_squared_error(mj["y_co2"][vj], q1j[vj])) / 2.5
                        + np.sqrt(mean_squared_error(mj["y_co"][vj], q2j[vj])) / 6.5
                    )
                if float(np.mean(losses)) < best_loss:
                    best_loss, best_g = float(np.mean(losses)), g_cand

            # \u82e5\u4e3a Continuous_DualModal\uff0c\u4fdd\u5b58\u5176\u4e25\u683c Inner-OOF \u9884\u6d4b\u4f9b\u5171\u5f62\u533a\u95f4\u6807\u5b9a\u4f7f\u7528
            if m_name == "Continuous_DualModal":
                for j in tr_idx:
                    sj = per_fold_streams[i][j]
                    mj = per_fold_maps[i][j]
                    p1_oof, p2_oof = inner_oof_zero[i][j]
                    inner_oof_dual[i][j] = apply_causal_smith_observer(
                        p1_oof, p2_oof, sj.obs_co2_causal, sj.obs_co_causal, mj["k_src"], sj.dt, interval_s, best_g
                    )

            si, mi = test_streams[i], test_maps[i]
            per_heat_smith.append(apply_causal_smith_observer(
                main_zero_preds[i][0], main_zero_preds[i][1],
                si.obs_co2_causal, si.obs_co_causal, mi["k_src"], si.dt, interval_s, best_g,
            ))
            per_heat_naive.append(apply_naive_lagged_sensor(si.raw_co2_zoh, si.raw_co_zoh, si.dt, interval_s))

        smith_preds[m_name] = per_heat_smith
        naive_preds[m_name] = per_heat_naive

    # 6. \u6c47\u603b\u5168\u6d88\u878d\u3001\u591a\u7089\u7ec4\u5408\u3001\u53cc\u6a21\u6001 Smith \u8f6f\u6d4b\u91cf\u4e0e Naive \u6ede\u540e\u4f20\u611f\u5668\u6301\u4e45\u6027\u57fa\u7ebf\u8868
    summary_rows, detail_rows = [], []
    all_eval = [(k, v, 100.0) for k, v in preds_map.items()]
    for m_name, interval_s in Q3_CFG.probe_intervals_s:
        if interval_s is None:
            continue
        saving = 0.0 if interval_s <= 0.5 else (1.0 - 0.5 / interval_s) * 100.0
        all_eval.append((f"Hybrid_Smith_{m_name}", smith_preds[m_name], saving))
    for m_name, interval_s in Q3_CFG.probe_intervals_s:
        if interval_s is None:
            continue
        saving = 0.0 if interval_s <= 0.5 else (1.0 - 0.5 / interval_s) * 100.0
        all_eval.append((f"Naive_Lagged_Sensor_{m_name}", naive_preds[m_name], saving))

    for name, plist, saving in all_eval:
        r2_1, r2_2, rm_1, rm_2 = [], [], [], []
        for i, (s, mp) in enumerate(zip(test_streams, test_maps)):
            v = mp["valid"]
            y1, y2 = mp["y_co2"][v], mp["y_co"][v]
            p1, p2 = plist[i][0][v], plist[i][1][v]
            s1, s2 = float(r2_score(y1, p1)), float(r2_score(y2, p2))
            e1, e2 = float(np.sqrt(mean_squared_error(y1, p1))), float(np.sqrt(mean_squared_error(y2, p2)))
            r2_1.append(s1)
            r2_2.append(s2)
            rm_1.append(e1)
            rm_2.append(e2)
            detail_rows.append({
                "Mode": name, "Heat": f"H{s.heat_id:02d}", "HeatNo": s.heat_no,
                "CO2_R2": s1, "CO2_RMSE": e1, "CO_R2": s2, "CO_RMSE": e2,
            })
        summary_rows.append({
            "Mode": name,
            "Probe_Save_pct": round(saving, 2),
            "CO2_Mean_R2": round(float(np.mean(r2_1)), 4),
            "CO2_Med_R2": round(float(np.median(r2_1)), 4),
            "CO2_Min_R2": round(float(np.min(r2_1)), 3),
            "CO2_RMSE": round(float(np.mean(rm_1)), 3),
            "CO_Mean_R2": round(float(np.mean(r2_2)), 4),
            "CO_Med_R2": round(float(np.median(r2_2)), 4),
            "CO_Min_R2": round(float(np.min(r2_2)), 3),
            "CO_RMSE": round(float(np.mean(rm_2)), 3),
        })

    df_sum = pd.DataFrame(summary_rows)
    df_det = pd.DataFrame(detail_rows)
    df_sum.to_csv(TABLE_DIR / "q3_loho_ablation_summary.csv", index=False)
    df_det.to_csv(TABLE_DIR / "q3_loho_per_heat_metrics.csv", index=False)

    # 7. \u9636\u6bb5\u8f6c\u6298\u70b9 (\xb130 s) \u4e0e\u771f\u5b9e\u67aa\u4f4d/\u4f9b\u6c27\u8df3\u53d8\u533a (Top10pct_dQ_Jump) \u8bef\u5dee\u5bf9\u6bd4 [P0-5 \u4fee\u590d\uff1a\u57fa\u4e8e\u53bb\u8865\u8d26\u8df3\u53d8\u540e\u7684\u771f\u5b9e q_causal]
    trans_rows = []
    for i, (s, r, mp) in enumerate(zip(test_streams, q2_recons, test_maps)):
        v = mp["valid"]
        tv = s.time_s[v]
        near = np.zeros(len(tv), dtype=bool)
        for tb in r.stage_t_breaks:
            near |= np.abs(tv - tb) <= 30.0
        dq = np.abs(np.gradient(s.q_causal[v], s.dt))
        jump = dq >= np.quantile(dq, 0.90)
        for reg_name, m in [("Breakpoint_pm30s", near), ("Top10pct_dQ_Jump", jump), ("Steady_Other", (~near) & (~jump))]:
            for m_tag in ("M0_ProcessOnly", "Combined_SoftGated_Hybrid"):
                p1, p2 = preds_map[m_tag][i][0][v], preds_map[m_tag][i][1][v]
                trans_rows.append({
                    "Heat": f"H{s.heat_id:02d}", "Region": reg_name, "Model": m_tag,
                    "CO2_RMSE": float(np.sqrt(mean_squared_error(mp["y_co2"][v][m], p1[m]))),
                    "CO_RMSE": float(np.sqrt(mean_squared_error(mp["y_co"][v][m], p2[m]))),
                })
    df_trans = pd.DataFrame(trans_rows).groupby(["Region", "Model"])[["CO2_RMSE", "CO_RMSE"]].mean().reset_index()
    df_trans.to_csv(TABLE_DIR / "q3_transition_regime_rmse.csv", index=False)

    # 8. [P0-6 \u6838\u5fc3\u4fee\u590d] \u4e25\u683c\u57fa\u4e8e Inner-OOF \u6b8b\u5dee\u4e0e\u6298\u5185\u8bad\u7ec3\u9636\u6bb5\u9608\u503c\u7684\u3010\u5c40\u90e8\u5c3a\u5ea6\u81ea\u9002\u5e94 90% \u5171\u5f62\u9884\u6d4b\u533a\u95f4\u3011
    #    \u5c3a\u5ea6\u51fd\u6570 \u03c3(t) = 0.45 * y_bar_tr + 0.55 * y_hat(t) \u81ea\u52a8\u9002\u914d\u65b0\u7248 H01 \u7b49\u9ad8\u6d53\u5ea6\u7089\u6b21\u7684\u5f02\u65b9\u5dee\u632f\u5e45\uff01
    dual_preds = smith_preds["Continuous_DualModal"]
    cov_rows, conf_hw_co2, conf_hw_co = [], [], []

    for label_c, test_plist, oof_source in [
        ("Hybrid_ZeroProbe", main_zero_preds, inner_oof_zero),
        ("Hybrid_Smith_Continuous", dual_preds, inner_oof_dual),
    ]:
        for i in range(10):
            tr_idx = [j for j in range(10) if j != i]
            # \u4e25\u683c\u7531\u6298\u5185 9 \u4e2a\u8bad\u7ec3\u7089\u6807\u5b9a 3 \u4e2a\u7269\u7406\u5439\u70bc\u9636\u6bb5\uff08\u8d77\u5439/\u722c\u5347/\u4e3b\u5439\u53ca\u672b\u6bb5\uff0c\u907f\u514d\u7a00\u758f\u672b\u6bb5\u5c16\u523a\u8272\u5757\uff01\uff09
            tr_breaks_frac = np.mean([q2_recons[j].stage_v_breaks[:2] / q2_recons[j].o2_cont[-1] for j in tr_idx], axis=0)
            stage_cuts = tr_breaks_frac * fold_o2_scales[i]
            y1_bar = float(np.mean(np.concatenate([per_fold_maps[i][j]["y_co2"][per_fold_maps[i][j]["valid"]] for j in tr_idx])))
            y2_bar = float(np.mean(np.concatenate([per_fold_maps[i][j]["y_co"][per_fold_maps[i][j]["valid"]] for j in tr_idx])))

            vi = test_maps[i]["valid"]
            st_te = np.searchsorted(stage_cuts, test_streams[i].o2_causal[vi])
            p1_te, p2_te = test_plist[i][0][vi], test_plist[i][1][vi]
            sc1_te = 0.55 * y1_bar + 0.45 * p1_te
            sc2_te = 0.55 * y2_bar + 0.45 * p2_te
            q1_hw = np.zeros(len(st_te))
            q2_hw = np.zeros(len(st_te))

            for stg in range(3):
                scores1_list, scores2_list = [], []
                for j in tr_idx:
                    vj = per_fold_maps[i][j]["valid"]
                    st_j = np.searchsorted(stage_cuts, per_fold_streams[i][j].o2_causal[vj])
                    mask_s = st_j == stg
                    if not mask_s.any():
                        continue
                    p1_j, p2_j = oof_source[i][j][0][vj][mask_s], oof_source[i][j][1][vj][mask_s]
                    y1_j = per_fold_maps[i][j]["y_co2"][vj][mask_s]
                    y2_j = per_fold_maps[i][j]["y_co"][vj][mask_s]
                    scores1_list.append(np.abs(y1_j - p1_j) / (0.55 * y1_bar + 0.45 * p1_j))
                    scores2_list.append(np.abs(y2_j - p2_j) / (0.55 * y2_bar + 0.45 * p2_j))

                pool1 = np.concatenate(scores1_list) if scores1_list else np.array([0.2])
                pool2 = np.concatenate(scores2_list) if scores2_list else np.array([0.2])
                lvl1 = min(1.0, np.ceil((len(pool1) + 1) * (1.0 - Q3_CFG.conformal_alpha)) / max(len(pool1), 1))
                lvl2 = min(1.0, np.ceil((len(pool2) + 1) * (1.0 - Q3_CFG.conformal_alpha)) / max(len(pool2), 1))
                m_te_stg = st_te == stg
                q1_hw[m_te_stg] = float(np.quantile(pool1, lvl1)) * sc1_te[m_te_stg]
                q2_hw[m_te_stg] = float(np.quantile(pool2, lvl2)) * sc2_te[m_te_stg]

            if label_c == "Hybrid_Smith_Continuous":
                conf_hw_co2.append(q1_hw)
                conf_hw_co.append(q2_hw)

            res1_i = test_maps[i]["y_co2"][vi] - p1_te
            res2_i = test_maps[i]["y_co"][vi] - p2_te
            cov_rows.append({
                "Model": label_c,
                "Heat": f"H{test_streams[i].heat_id:02d}",
                "CO2_Coverage": float(np.mean(np.abs(res1_i) <= q1_hw)),
                "CO2_HalfWidth": float(np.mean(q1_hw)),
                "CO_Coverage": float(np.mean(np.abs(res2_i) <= q2_hw)),
                "CO_HalfWidth": float(np.mean(q2_hw)),
            })
    df_cov = pd.DataFrame(cov_rows)
    df_cov.to_csv(TABLE_DIR / "q3_conformal_90pct_coverage.csv", index=False)

    # 9. \u5168\u6ce2\u6bb5 PLS VIP \u6ce2\u957f\u91cd\u8981\u5ea6\u89e3\u91ca\u5206\u6790
    common_wl = np.linspace(PHYSICS.common_wl_min, PHYSICS.common_wl_max, PHYSICS.common_wl_points)
    x_vip = StandardScaler().fit_transform(np.vstack([s.spec_prob[m["valid"]] for s, m in zip(test_streams, test_maps)]))
    vip_dict = {"Wavelength_nm": common_wl}
    for gname, key in [("CO2_VIP", "y_co2"), ("CO_VIP", "y_co")]:
        yvec = np.concatenate([m[key][m["valid"]] for m in test_maps])
        pls = PLSRegression(n_components=8).fit(x_vip, yvec)
        t_s, w_s, q_s = pls.x_scores_, pls.x_weights_, pls.y_loadings_
        ss = np.sum(t_s ** 2, axis=0) * (q_s[0] ** 2)
        vip_dict[gname] = np.sqrt(x_vip.shape[1] * ((w_s ** 2) / np.sum(w_s ** 2, axis=0)) @ ss / (ss.sum() + 1e-12))
    df_vip = pd.DataFrame(vip_dict)
    df_vip.to_csv(TABLE_DIR / "q3_pls_vip_wavelength.csv", index=False)

    if make_plots:
        _plot_q3_diagnostics(transfer_mat, labels, df_sum, df_vip, test_streams)
        _plot_q3_all10_predictions(test_streams, test_maps, main_zero_preds, dual_preds, conf_hw_co2, conf_hw_co)

    return {
        "transfer": df_transfer,
        "summary": df_sum,
        "detail": df_det,
        "transition": df_trans,
        "conformal": df_cov,
        "fold_delays": pd.DataFrame(fold_delay_rows),
    }


def _plot_q3_diagnostics(
    mat: np.ndarray,
    labels: List[str],
    df_sum: pd.DataFrame,
    df_vip: pd.DataFrame,
    streams: List[CausalOnlineStreams],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.3))
    im = axes[0].imshow(np.clip(mat, -0.5, 1.0), cmap="RdYlGn", vmin=-0.5, vmax=1.0)
    short = [l.split("(")[0] for l in labels]
    axes[0].set_xticks(range(10))
    axes[0].set_yticks(range(10))
    axes[0].set_xticklabels(short, rotation=45, fontsize=8)
    axes[0].set_yticklabels(short, fontsize=8)
    for i in range(10):
        for j in range(10):
            axes[0].text(
                j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.7,
                color="black" if mat[i, j] > 0.1 else "white",
            )
    axes[0].set_title("10\xd710 Single-Heat Transfer Matrix (Mean R\xb2)")
    axes[0].set_xlabel("Test Heat")
    axes[0].set_ylabel("Training Heat")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: [P1-1 \u4fee\u590d] \u53cc\u6a21\u6001\u8f6f\u6d4b\u91cf Pareto \u66f2\u7ebf vs \u672a\u8865\u507f\u6ede\u540e\u786c\u4ef6\u5206\u6790\u4eea\u57fa\u7ebf\u540c\u56fe\u6b63\u9762\u4ea4\u950b\uff01
    pareto_smith = [
        ("Combined_SoftGated_Hybrid", "0-Probe\n(100%)"),
        ("Hybrid_Smith_Pulse_120s_99.6pctSave", "120s\n(99.6%)"),
        ("Hybrid_Smith_Pulse_60s_99.2pctSave", "60s\n(99.2%)"),
        ("Hybrid_Smith_Pulse_30s_98.3pctSave", "30s\n(98.3%)"),
        ("Hybrid_Smith_Pulse_15s_96.7pctSave", "15s\n(96.7%)"),
        ("Hybrid_Smith_Continuous_DualModal", "Cont.\n(0.5s)"),
    ]
    pareto_naive = [
        "Naive_Lagged_Sensor_Pulse_120s_99.6pctSave",
        "Naive_Lagged_Sensor_Pulse_60s_99.2pctSave",
        "Naive_Lagged_Sensor_Pulse_30s_98.3pctSave",
        "Naive_Lagged_Sensor_Pulse_15s_96.7pctSave",
        "Naive_Lagged_Sensor_Continuous_DualModal",
    ]
    idx_df = df_sum.set_index("Mode")
    sub_s = idx_df.loc[[p[0] for p in pareto_smith]]
    sub_n = idx_df.loc[pareto_naive]
    xi = np.arange(len(pareto_smith))
    xi_n = np.arange(1, len(pareto_smith))

    axes[1].plot(xi, sub_s["CO2_Mean_R2"], "o-", lw=2.1, ms=6.2, color="#1f77b4", label="CO\u2082 Dual-Modal Hybrid (Ours)")
    axes[1].plot(xi, sub_s["CO_Mean_R2"], "s-", lw=2.1, ms=6.2, color="#d62728", label="CO Dual-Modal Hybrid (Ours)")
    axes[1].plot(xi_n, np.clip(sub_n["CO2_Mean_R2"], -0.2, 1.0), "o--", lw=1.3, ms=4.8, color="#1f77b4", alpha=0.55,
                 label="CO\u2082 Naive Lagged Sensor")
    axes[1].plot(xi_n, np.clip(sub_n["CO_Mean_R2"], -0.2, 1.0), "s--", lw=1.3, ms=4.8, color="#d62728", alpha=0.55,
                 label="CO Naive Lagged Sensor")
    axes[1].set_xticks(xi)
    axes[1].set_xticklabels([p[1] for p in pareto_smith], fontsize=8)
    axes[1].set_ylim(-0.05, 1.02)
    axes[1].set_ylabel("Strict Nested-LOHO Mean R\xb2")
    axes[1].set_title("Accuracy vs. Probe Saving (Ours vs. Lagged Sensor)")
    axes[1].legend(fontsize=7.3, loc="lower right")

    axes[2].plot(df_vip["Wavelength_nm"], df_vip["CO2_VIP"], color="#1f77b4", lw=1.4, label="CO\u2082 VIP")
    axes[2].plot(df_vip["Wavelength_nm"], df_vip["CO_VIP"], color="#d62728", lw=1.4, label="CO VIP")
    axes[2].axhline(1.0, color="k", ls="--", lw=0.9, label="VIP = 1 Threshold")
    ymax_v = max(float(df_vip[["CO2_VIP", "CO_VIP"]].to_numpy().max()) * 1.08, 1.5)
    axes[2].set_ylim(0.0, ymax_v)
    for wv, tag in [(589.3, "Na I"), (766.5, "K I")]:
        axes[2].axvline(wv, color="0.45", ls=":", lw=1.0)
        axes[2].text(wv + 5, ymax_v * 0.90, tag, fontsize=8.5, color="0.25")
    axes[2].set_xlabel("Wavelength \u03bb (nm)")
    axes[2].set_ylabel("PLS VIP Score")
    axes[2].set_title("Spectral Wavelength Importance (PLS VIP)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, "q3_transfer_pareto_and_vip")
    plt.close(fig)

    # [P1-5 & P0-6 \u4fee\u590d] \u4e25\u683c\u4ec5\u4f7f\u7528\u5f00\u5439\u524d 60 s \u56e0\u679c\u5de5\u51b5\u6307\u7eb9\u805a\u7c7b\uff0c\u62d3\u5bbd\u753b\u5e03\u5e76\u65cb\u8f6c\u6807\u7b7e\uff0c\u5f7b\u5e95\u6d88\u9664\u6587\u5b57\u78b0\u649e\u4e0e\u4e8b\u540e\u6307\u6807\u6cc4\u6f0f\uff01
    desc_c60 = np.vstack([s.desc_causal60 for s in streams])
    z_link = linkage(StandardScaler().fit_transform(desc_c60), method="ward")
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    dendrogram(
        z_link,
        labels=[f"H{s.heat_id:02d}: {s.heat_no} ({s.converter})" for s in streams],
        ax=ax,
        leaf_rotation=18,
        leaf_font_size=8.8,
        color_threshold=0.7 * z_link[:, 2].max(),
    )
    ax.set_ylabel("Ward Linkage Distance")
    ax.set_title("Causal First-60s Spectral & Flow Hierarchical Clustering (All 10 Heats, Zero Future Leakage)")
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, "q3_heat_cluster_dendrogram")
    plt.close(fig)


def _plot_q3_all10_predictions(
    streams: List[CausalOnlineStreams],
    test_maps: List[Dict[str, np.ndarray]],
    zero_preds: List[Tuple[np.ndarray, np.ndarray]],
    dual_preds: List[Tuple[np.ndarray, np.ndarray]],
    hw_co2: List[np.ndarray],
    hw_co: List[np.ndarray],
) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(16, 17.5))
    for idx, (ax, s, mp, zp, dp, h1, h2) in enumerate(
        zip(axes.flat, streams, test_maps, zero_preds, dual_preds, hw_co2, hw_co)
    ):
        v = mp["valid"]
        tm = s.time_s[v] / 60.0
        p2, p1 = dp[1][v], dp[0][v]
        ax.fill_between(tm, np.clip(p2 - h2, 0, None), p2 + h2, color="#d62728", alpha=0.15, lw=0, label="90% Adaptive Conformal CI (CO)")
        ax.fill_between(tm, np.clip(p1 - h1, 0, None), p1 + h1, color="#2ca02c", alpha=0.17, lw=0, label="90% Adaptive Conformal CI (CO\u2082)")
        ax.plot(tm, mp["y_co"][v], color="#111111", lw=1.5, label="True Real-Time CO")
        ax.plot(tm, p2, color="#d62728", lw=1.2, ls="--", label="Pred CO (Dual-Modal)")
        ax.plot(tm, zp[1][v], color="#ff9896", lw=0.90, alpha=0.85, label="Pred CO (Zero-Probe Hybrid)")
        ax.plot(tm, mp["y_co2"][v], color="#1f77b4", lw=1.4, label="True Real-Time CO\u2082")
        ax.plot(tm, p1, color="#2ca02c", lw=1.2, ls="--", label="Pred CO\u2082 (Dual-Modal)")

        r2_c2 = r2_score(mp["y_co2"][v], p1)
        r2_co = r2_score(mp["y_co"][v], p2)
        ax.set_title(f"H{s.heat_id:02d} ({s.heat_no}) | Strict Nested-LOHO R\xb2: CO\u2082={r2_c2:.3f}, CO={r2_co:.3f}", fontsize=9.7)
        ax.set_xlabel("Real-Time Blowing Time (min)", fontsize=8.5)
        ax.set_ylabel("Off-Gas Conc. (vol%)", fontsize=8.5)
        if idx == 0:
            ax.legend(loc="upper left", ncol=2, fontsize=6.7)

    fig.suptitle("Strict Nested-LOHO Real-Time Prediction & Locally-Scaled 90% Conformal Intervals (All 10 Heats)", fontsize=13.4, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save_figure(fig, "q3_all10_loho_predictions")
    plt.close(fig)
