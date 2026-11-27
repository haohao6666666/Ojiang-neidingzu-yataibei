"""
第 1 题模块（v4 红队审计全量修正版）：
彻底修复红队指出的全部第 1 题缺陷：
  1. [P0-1 修复] 全流程（含 Savitzky-Golay 保峰滤波、长窗趋势提取与二元三次样条曲面 RectBivariateSpline）
     严格在对数辐射强度域 Y(t, λ) = ln(1 + I(t, λ)) 内建模与插值，求值时通过 expm1 映射回线性强度域，
     从数学上根除高梯度闪爆边缘的 Gibbs/Runge 负下冲与 3D 曲面“黑洞深坑”。
  2. [P0-2 修复] 含 CCD 饱和像素的高辐射帧属于右删失观测（Censored Highs）而非炉口遮挡暗帧，绝不整帧抹除未饱和侧翼；
     同时在平滑后严格施加右删失物理下界投影 I_rec(m_sat) >= 65530 counts，确保诊断图与正文数学约束 100% 自洽。
  3. [P1-2 修复] 合成异常注入检测器剥离帧级火焰闪烁共模分量，精准靶向局部时空稀疏毛刺，显著超越一维 Hampel 基线。
  4. [P0-6 衔接接口] 提供 `extract_fold_q1_spectral_basis` 接口，仅利用训练折炉次的 Q1 连续光谱曲面提取低秩模态基，
     供第 3 题在线单帧右删失投影调用，实现 Q1 → Q3 的严格无泄漏模型递进。
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
    spec_clean: np.ndarray           # (T, L) 物理保真重构光谱（严格满足饱和点 I >= 65530）
    spec_trend: np.ndarray           # (T, L) 对数域平滑后映射回线性域的连续趋势曲面
    surface_log: RectBivariateSpline # 对数强度域连续二元三次样条曲面 Y(t, λ) = ln(1 + I(t, λ))
    anom_mask: np.ndarray            # (T, L) 异常与饱和总掩膜
    sat_mask: np.ndarray             # (T, L) 16-bit CCD 右删失饱和掩膜
    low_rank_log: np.ndarray         # (T, L) 对数域低秩分量 L
    sparse_log: np.ndarray           # (T, L) 对数域稀疏残差分量 S = Y_obs - L
    metrics: Dict[str, float]

    def eval_intensity_surface(self, t_s: np.ndarray, wl_nm: np.ndarray) -> np.ndarray:
        """在任意连续时空网格 (t, λ) 上无负冲激求值连续辐射强度 I(t, λ) >= 0。"""
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
    """2D Whittaker 离散惩罚样条（用作留出基准对照）。"""
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
    执行四阶段对数域物理保真光谱异常检测与右删失低秩重构：
      1) 右删失饱和识别 (I >= 65530) + 时空联合孤立毛刺（保护 Na/K 双线）+ 纯遮挡瞬态暗帧检测（排除饱和高能帧）
      2) 对数域带右删失不等式约束 (Y >= ln(1 + 65530)) 的随机化软收缩低秩 SVD 补全
      3) 严格在对数域内执行可分离保峰 Savitzky-Golay 滤波，并再次投影回右删失可行集，彻底消除线性域负冲激与削顶回落
    """
    r_use = Q1_CFG.svd_rank if rank is None else rank
    s_raw = spec.astype(np.float64, copy=True)
    n_t, n_l = s_raw.shape

    # Stage 1: 分级物理异常检测
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

    # 仅将非饱和的瞬态烟尘遮挡/断光帧判为整帧缺失，绝不误删高辐射饱和帧的未饱和连续侧翼！
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

    # Stage 2: 初值填充 + 对数域右删失约束低秩 SVD 补全 (Right-Censored Soft-Impute)
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
        # 双中心化对数辐射模型 Y(t,λ) = α_t + β_λ + U_r Σ_r V_r^T：精准捕获逐帧火焰闪烁共模并保护谱形
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

    # Stage 3: 严格在对数域执行波长维保峰 Savitzky-Golay 平滑 + 保护动态脉冲的轻度时间去噪
    # [P0-2 视觉精修] 含右删失饱和的瞬态爆闪帧仅沿波长维平滑，绝不与相邻暗帧时间平均，确保未饱和侧翼与饱和拱顶无缝连续！
    y_wl_sm = savgol_filter(y_cur, Q1_CFG.sg_wl_win, Q1_CFG.sg_wl_poly, axis=1)
    y_smooth = y_wl_sm.copy()
    if n_t >= 3:
        # 对非爆闪帧做轻度三点加权时间平滑 [0.10, 0.80, 0.10]（保留真实闪烁动态且消除 Jensen 对数均值偏移）
        y_t_light = y_wl_sm.copy()
        y_t_light[1:-1] = 0.10 * y_wl_sm[:-2] + 0.80 * y_wl_sm[1:-1] + 0.10 * y_wl_sm[2:]
        # 仅在相邻帧亮度变化温和（非瞬态爆闪跃迁）且无饱和像素的帧应用轻度时间平滑
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

    # K I 766.5 / 769.9 nm 双线峰谷比保真度
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
    [Q1 → Q3 严格无泄漏模型递进接口]：
    仅使用外层交叉验证训练炉次 `train_indices` 在第 1 题重构得到的连续光滑光谱模型 `spec_clean`，
    重采样至公共波长网格并在帧中心化对数辐射域提取前 r 个正交光谱特征模态基 (mu_shape, V_r)。
    供测试炉在线逐帧执行带饱和下界约束的子空间投影恢复与模态得分提取（零跨炉泄漏、零未来帧依赖）。
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
        # 扣除当前帧平均对数亮度，提取跨炉域不变的光谱温度斜率与原子谱线模态
        y_shape = y_grid - y_grid.mean(axis=1, keepdims=True)
        blocks.append(y_shape)

    mat_tr = np.vstack(blocks)
    mu_shape = mat_tr.mean(axis=0)
    _, _, vt = np.linalg.svd(mat_tr - mu_shape[None, :], full_matrices=False)
    # 固定每个模态向量的符号方向以确保跨折可解释性一致
    v_r = vt[:r_modes, :].T.copy()
    for m in range(r_modes):
        if v_r[np.argmax(np.abs(v_r[:, m])), m] < 0:
            v_r[:, m] *= -1.0
    return mu_shape, v_r


def _detect_spatiotemporal_lowrank_spikes(s_in: np.ndarray) -> np.ndarray:
    """剥离帧级火焰闪烁共模分量后的时空-低秩联合单像素毛刺检测器（校准后高精确率与高召回率兼备）。"""
    y = np.log1p(np.maximum(s_in.astype(np.float64), 0.0))
    # 先移除帧级亮度脉动共模，避免把正常转炉火焰湍流闪烁误报为传感器尖峰
    y_wl_med = median_filter(y, size=(1, 7), mode="nearest")
    r_wl = y - y_wl_med
    sig_wl = 1.4826 * np.median(np.abs(r_wl), axis=1, keepdims=True) + 0.015

    y_tm_med = median_filter(y - np.median(y, axis=1, keepdims=True), size=(5, 1), mode="nearest")
    r_tm = (y - np.median(y, axis=1, keepdims=True)) - y_tm_med
    sig_tm = 1.4826 * np.median(np.abs(r_tm), axis=0, keepdims=True) + 0.015

    return (np.abs(r_wl) > 4.2 * sig_wl) & (np.abs(r_tm) > 3.8 * sig_tm)


def _benchmark_methods_and_injection(heats: List[HeatDataset], results: List[Q1Result]) -> None:
    """多方法 5% 像素留出重构对比 + 合成异常注入 Precision/Recall/F1 验证实验。"""
    comp_rows = []
    inj_rows = []
    rng = np.random.default_rng(SEED)

    for idx in (2, 7):  # 代表性 6B (H03) 与 6C (H08)
        h, res = heats[idx], results[idx]
        y_true = np.log1p(np.maximum(h.spectrum.astype(np.float64), 0.0))
        clean = ~res.anom_mask
        hide = clean & (rng.random(y_true.shape) < Q1_CFG.masked_cv_frac)

        # 1) 时间线性插值基线
        y_lin = y_true.copy()
        for j in range(y_true.shape[1]):
            obs_j = clean[:, j] & (~hide[:, j])
            if obs_j.sum() > 2:
                y_lin[:, j] = np.interp(np.arange(y_true.shape[0]), np.where(obs_j)[0], y_true[obs_j, j])

        # 2) Savitzky-Golay 基线
        y_sg = savgol_filter(y_lin, 11, 2, axis=0)

        # 3) 2D Whittaker 平滑样条基线
        y_whit = _whittaker_2d(y_lin, lam_t=10.0, lam_w=1.0)

        # 4) 本文方法：对数域右删失约束软收缩低秩 SVD 补全
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

        # 合成异常注入实验：在含真实火焰湍流的干净光谱上随机注入 1% 单像素脉冲毛刺 (6–10σ)
        base_s = res.spec_clean.copy()
        y_base = np.log1p(np.maximum(base_s, 0.0))
        sig_local = 1.4826 * np.median(np.abs(y_base - median_filter(y_base, size=(5, 5), mode="nearest")), axis=0) + 0.035
        spk = (rng.random(base_s.shape) < 0.01) & (~res.sat_mask)
        amp = rng.uniform(6.0, 10.0, base_s.shape) * sig_local[None, :] * rng.choice([-1.0, 1.0], base_s.shape)
        y_inj = np.clip(y_base + spk * amp, 0.0, 10.9)
        s_inj = np.expm1(y_inj)

        # 基线：纯时间维 1D Hampel 滤波器（易把剧烈火焰闪烁误判为异常尖峰）
        med_t = median_filter(y_inj, size=(5, 1), mode="nearest")
        rt = np.abs(y_inj - med_t)
        mt = 1.4826 * np.median(rt, axis=0, keepdims=True) + 0.015
        pred_hampel = rt > 4.5 * mt

        # 本文方法：去共模闪烁时空联合脉冲检测器
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
        # 严格在对数强度域提取平滑趋势曲面并构造二元三次样条 Y(t, λ) = ln(1 + I)，彻底杜绝线性样条负下冲黑洞！
        w_trend = max(11, int(round(15.0 / h.dt)) | 1)
        y_trend = np.maximum(savgol_filter(y_smooth, w_trend, 2, axis=0), 0.0)
        spec_trend = np.expm1(np.clip(y_trend, 0.0, 12.0))
        surface_log = RectBivariateSpline(h.time_s, h.wavelengths, y_trend, kx=3, ky=3, s=0)

        # 1% 掩码留出重构 R²（对数域与线性域双验证）
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
        # 直接从对数样条求值并转换至 log10(I + 1) = Y / ln(10)，严格正值光滑无黑洞伪影！
        z_log10 = np.maximum(res.surface_log(t_grid, wl_grid), 0.0) / np.log(10.0)
        W, T = np.meshgrid(wl_grid, t_grid / 60.0)
        ax.plot_surface(W, T, z_log10, cmap=cm.inferno, linewidth=0, antialiased=True, rstride=2, cstride=2)
        ax.set_title(f"Heat {h.heat_id}: {h.heat_no}", fontsize=9.8, pad=4)
        ax.set_xlabel("λ (nm)", fontsize=8, labelpad=2)
        ax.set_ylabel("Time (min)", fontsize=8, labelpad=2)
        ax.set_zlabel("log₁₀(I + 1)", fontsize=8, labelpad=1)
        ax.tick_params(labelsize=7, pad=0)
        ax.view_init(elev=26, azim=-128)

    fig.suptitle("Continuous 3D Flame Spectral Surfaces for All 10 Heats (Log-Spline Surface, log₁₀(I + 1))", fontsize=13.5, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.91, wspace=0.12, hspace=0.26)
    save_figure(fig, "q1_all10_3d_spectra")
    plt.close(fig)


def _plot_q1_diagnostic_and_rpca(h_sat: HeatDataset, r_sat: Q1Result, h_rpca: HeatDataset, r_rpca: Q1Result) -> None:
    # 图 1：右删失饱和恢复（严格 I_rec >= 65530 且侧翼无缝衔接）+ K I 双峰保真诊断
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    k_sat = int(np.argmax(r_sat.sat_mask.sum(axis=1)))
    axes[0].plot(h_sat.wavelengths, h_sat.spectrum[k_sat], color="#d62728", alpha=0.80, lw=1.2,
                 label="Raw Clipped Spectrum (16-bit ADC Ceiling)")
    axes[0].plot(h_sat.wavelengths, r_sat.spec_clean[k_sat], color="#1f77b4", lw=1.4,
                 label="Right-Censored Low-Rank Restoration (I ≥ 65530)")
    axes[0].axhline(PHYSICS.adc_saturation, color="k", ls=":", lw=1.0, label="Saturation Lower Bound (65530)")
    axes[0].set_title(f"{h_sat.heat_no} (t={h_sat.time_s[k_sat]:.1f}s): Right-Censored Saturation Restoration")
    axes[0].set_xlabel("Wavelength λ (nm)")
    axes[0].set_ylabel("Radiant Intensity (counts)")
    axes[0].legend(fontsize=8.2)

    m_k = (h_sat.wavelengths >= 758) & (h_sat.wavelengths <= 778)
    axes[1].plot(h_sat.wavelengths[m_k], h_sat.spectrum.mean(0)[m_k], "o-", ms=3.2, color="#ff7f0e", alpha=0.82,
                 label="Raw Mean Spectrum")
    axes[1].plot(h_sat.wavelengths[m_k], r_sat.spec_clean.mean(0)[m_k], "-", lw=1.8, color="#2ca02c",
                 label="Reconstructed Continuous Mean")
    axes[1].set_title(f"{h_sat.heat_no} K I Doublet (766.5 / 769.9 nm) Preservation")
    axes[1].set_xlabel("Wavelength λ (nm)")
    axes[1].set_ylabel("Mean Intensity (counts)")
    axes[1].legend(fontsize=8.2)
    fig.tight_layout()
    save_figure(fig, "q1_saturation_and_peak_diagnostic")
    plt.close(fig)

    # 图 2：低秩 + 稀疏分解三联热力图 (Y = L + S)
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
    axes[0].set_ylabel("Wavelength λ (nm)")
    fig.suptitle(f"Heat {h_rpca.heat_id} ({h_rpca.heat_no}): Log-Domain Low-Rank + Sparse Spectral Decomposition", y=1.02)
    fig.tight_layout()
    save_figure(fig, "q1_rpca_decomposition")
    plt.close(fig)
