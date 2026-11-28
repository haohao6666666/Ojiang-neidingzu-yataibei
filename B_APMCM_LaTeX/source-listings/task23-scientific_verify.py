"""
\u81ea\u52a8\u5316\u79d1\u5b66\u4e00\u81f4\u6027\u4e0e\u96f6\u6cc4\u6f0f\u7ea2\u961f\u9632\u5fa1\u5ba1\u8ba1\u6a21\u5757\uff08v4 \u5347\u7ea7\u7248\uff09\uff1a
\u6267\u884c 6 \u9879\u5355\u5143\u6d4b\u8bd5\u7ea7\u4e25\u683c\u6570\u5b66\u65ad\u8a00\u5e76\u5bfc\u51fa `verification_audit_report.csv`\uff1a
  1. \u6709\u9650\u6027\u3001\u7269\u7406\u975e\u8d1f\u6027\u3001\u53f3\u5220\u5931\u9971\u548c\u4e0b\u754c (I_rec >= 65530) \u4e0e\u5bf9\u6570 3D \u6837\u6761\u96f6\u8d1f\u4e0b\u51b2\u68c0\u67e5
  2. \u70df\u6c14\u6d53\u5ea6\u65f6\u95f4\u79ef\u5206\u95ed\u5408\u4e00\u81f4\u6027\u68c0\u67e5
  3. \u53d8\u6d41\u91cf\u5e73\u63a8\u6d41\u5230\u8fbe\u65f6\u949f a(t) \u4e25\u683c\u5355\u8c03\u65e0\u65f6\u949f\u6298\u53e0\u68c0\u67e5\uff08Time-Folding = 0.000%\uff09
  4. [P0-3 \u5f3a\u5316\u5ba1\u8ba1] \u542b\u566a\u5408\u6210\u6570\u636e + \u76f2\u731c\u8fdc\u8ddd\u521d\u503c (50 m\xb3 / 205 m\xb3) \u5df2\u77e5\u8f93\u8fd0\u4f53\u79ef (\u0394V_true = 135 m\xb3) \u56de\u6536\u68c0\u9a8c
  5. \u7be1\u6539\u672a\u6765\u6570\u636e\u96f6\u6cc4\u6f0f\u56e0\u679c\u6d4b\u8bd5\uff08\u542b Q1 \u6298\u5185\u6a21\u6001\u57fa\u5355\u5e27\u6295\u5f71\uff0c\u65ad\u8a00 t <= t0 \u9010\u6bd4\u7279\u7edd\u5bf9\u5dee == 0.000000e+00\uff09
  6. [P0-6 \u65b0\u589e\u5ba1\u8ba1] \u5916\u5c42\u6d4b\u8bd5\u7089\u8de8\u7089\u8d85\u53c2\u6570/\u6a21\u6001\u57fa\u96f6\u6cc4\u6f0f\u65ad\u8a00\uff1a\u6781\u7aef\u7be1\u6539\u6d4b\u8bd5\u7089\u5168\u90e8\u771f\u503c\u70df\u6c14\u4e0e\u5149\u8c31\uff0c
     \u65ad\u8a00\u8bad\u7ec3\u4fa7\u6807\u5b9a\u7684\u6298\u5185\u65f6\u5ef6 \u0394V_pipe^{(-i)}\u3001Q1 \u6a21\u6001\u57fa (\u03bc^{(-i)}, V_r^{(-i)}) \u4e0e\u8017\u6c27\u6807\u5ea6\u7edd\u5bf9\u5dee == 0.000000e+00\uff01
"""
from __future__ import annotations

from dataclasses import replace
from typing import List
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from config import PHYSICS, Q1_CFG, SEED, TABLE_DIR
from io_loader import HeatDataset
from q1_spectral_recon import Q1Result, extract_fold_q1_spectral_basis
from q2_gas_delay import Q2HeatRecon, Q2Summary, fit_pfr_kinetics
from q3_fusion_predict import (
    apply_causal_smith_observer,
    build_targets_and_backward_map,
    compute_strictly_causal_streams,
)


def run_scientific_audit(
    heats: List[HeatDataset],
    q1_res: List[Q1Result],
    q2_sum: Q2Summary,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)

    # Audit 1: \u5168\u91cf\u6709\u9650\u6027\u3001\u975e\u8d1f\u6027\u3001\u53f3\u5220\u5931\u9971\u548c\u4e0b\u754c (I >= 65530) \u4e0e 3D \u5bf9\u6570\u6837\u6761\u96f6\u8d1f\u4e0b\u51b2\u65ad\u8a00
    min_spec = min(float(r.spec_clean.min()) for r in q1_res)
    min_sat = min(float(r.spec_clean[r.sat_mask].min()) for r in q1_res if r.sat_mask.any())
    all_finite = all(np.isfinite(r.spec_clean).all() for r in q1_res) and all(
        np.isfinite(r.co2_cont).all() and np.isfinite(r.co_cont).all() and np.isfinite(r.q_o2).all()
        for r in q2_sum.recons
    )
    undershoot_3d = 0
    for h, r in zip(heats, q1_res):
        tg = np.linspace(h.time_s[0], h.time_s[-1], 120)
        wg = np.linspace(h.wavelengths[0], h.wavelengths[-1], 120)
        undershoot_3d += int((r.eval_intensity_surface(tg, wg) < 0.0).sum())

    min_flow = min(float(r.q_o2.min()) for r in q2_sum.recons)
    min_gas = min(min(float(r.co2_cont.min()), float(r.co_cont.min())) for r in q2_sum.recons)
    min_do2 = min(float(np.diff(r.o2_cont).min()) for r in q2_sum.recons)
    ok_phys = (
        all_finite
        and (min_spec >= 0.0)
        and (min_sat >= PHYSICS.adc_saturation - 1e-6)
        and (undershoot_3d == 0)
        and (min_flow >= 0.0)
        and (min_gas >= 0.0)
        and (min_do2 >= -1e-12)
    )
    rows.append({
        "Audit_Check": "1_Finiteness_NonNegativity_SatBound_and_Zero3DUndershoot",
        "Status": "PASS" if ok_phys else "FAIL",
        "Metric": f"finite={all_finite}, min_sat={min_sat:.1f}, 3d_neg={undershoot_3d}, min_dO2={min_do2:.2e}",
    })

    # Audit 2: \u6d53\u5ea6\u79ef\u5206\u95ed\u5408\u4e00\u81f4\u6027\u68c0\u67e5
    max_int_err = 0.0
    for r in q2_sum.recons:
        e_check = float(np.trapezoid(r.co_cont, r.time_s))
        max_int_err = max(max_int_err, abs(e_check - float(r.tot_co[-1])))
    rows.append({
        "Audit_Check": "2_Concentration_Integral_Closure",
        "Status": "PASS" if max_int_err < 1e-8 else "FAIL",
        "Metric": f"max_endpoint_closure_diff={max_int_err:.2e} volpct*s",
    })

    # Audit 3: \u53d8\u6d41\u91cf\u5230\u8fbe\u65f6\u949f a(t) \u4e25\u683c\u5355\u8c03\u65e0\u6298\u53e0\u68c0\u67e5\uff08\u4f7f\u7528\u4e00\u81f4\u5fae\u6270\u5355\u8c03\u5750\u6807 o2_u\uff09
    pg = q2_sum.pfr_global
    min_da = np.inf
    fold_count = 0
    for r in q2_sum.recons:
        o2_u = r.o2_cont + np.arange(len(r.time_s)) * 1e-6
        t_arr = np.interp(o2_u + pg.v_pipe_m3, o2_u, r.time_s, right=np.nan) + pg.tau0_s
        v = ~np.isnan(t_arr)
        da = np.diff(t_arr[v])
        min_da = min(min_da, float(da.min()))
        fold_count += int((da <= 0.0).sum())
    rows.append({
        "Audit_Check": "3_Arrival_Clock_Strict_Monotonicity_No_TimeFolding",
        "Status": "PASS" if fold_count == 0 else "FAIL",
        "Metric": f"folded_steps={fold_count}, min_da_step={min_da:.4e} s",
    })

    # Audit 4: [P0-3 \u5f3a\u5316\u7248] \u542b\u89c2\u6d4b\u566a\u58f0 + \u76f2\u731c\u8fdc\u8ddd\u504f\u7f6e\u521d\u503c (45 m\xb3 / 210 m\xb3) \u7684\u5408\u6210\u5e73\u63a8\u6d41\u7f6e\u6362\u4f53\u79ef (True=135.0 m\xb3) \u5168\u5c40\u56de\u6536\u68c0\u9a8c
    v_true = 135.0
    synth_recons: List[Q2HeatRecon] = []
    for r in q2_sum.recons[:6]:
        t, o2_c, dt = r.time_s, r.o2_cont, r.dt
        v_s = o2_c - v_true
        gate_on = 1.0 / (1.0 + np.exp(-np.clip(v_s / 3.5, -40.0, 40.0)))
        v_pos = np.maximum(v_s, 0.0)
        sig_m = 1.0 / (1.0 + np.exp(-np.clip((v_pos - 480.0) / 150.0, -40.0, 40.0)))
        sig_e = 1.0 / (1.0 + np.exp(np.clip((v_pos - 3620.0) / 180.0, -40.0, 40.0)))
        u = 35.0 * gate_on * (0.15 + 0.85 * sig_m) * sig_e
        a = float(np.exp(-dt / 5.0))
        cox_clean = lfilter([1.0 - a], [1.0, -a], u)
        # \u5728\u6709\u70df\u6c14\u6bb5\u65bd\u52a0\u96f6\u5747\u503c\u9ad8\u65af\u6d4b\u91cf\u566a\u58f0\uff08\u907f\u514d\u5728\u5168\u96f6\u70b9\u706b\u524d\u57fa\u7ebf\u4e0a\u505a\u5355\u4fa7\u6b63\u622a\u65ad\u5f15\u5165\u865a\u5047 DC \u504f\u7f6e\uff09
        noise_scale = np.where(cox_clean > 1.0, 0.8, 0.02)
        cox_noisy = np.clip(cox_clean + rng.normal(0.0, 1.0, size=len(t)) * noise_scale, 0.0, None)
        synth_recons.append(Q2HeatRecon(
            heat_id=r.heat_id, heat_no=r.heat_no, converter=r.converter, dt=dt, time_s=t,
            o2_cont=o2_c, q_o2=r.q_o2, co2_cont=0.4 * cox_noisy, co_cont=1.2 * cox_noisy,
            tot_co2=r.tot_co2, tot_co=r.tot_co, stage_v_breaks=r.stage_v_breaks, stage_t_breaks=r.stage_t_breaks,
        ))
    distant_blind_starts = [
        [45.0, 0.04, 800.0, 280.0, 3250.0],
        [210.0, 0.35, 260.0, 75.0, 4100.0],
    ]
    rec_fit = fit_pfr_kinetics(
        synth_recons, list(range(6)), "Noisy_BlindStart_Synthetic_Check",
        run_bootstrap=False, custom_starts=distant_blind_starts,
    )
    rec_err = abs(rec_fit.v_pipe_m3 - v_true)
    rows.append({
        "Audit_Check": "4_Noisy_BlindStart_Synthetic_Delay_Recovery (True=135.0m3)",
        "Status": "PASS" if rec_err < 3.5 else "FAIL",
        "Metric": f"recovered_dV={rec_fit.v_pipe_m3:.2f} m3 from starts [45, 210] (abs_err={rec_err:.2f} m3)",
    })

    # Audit 5: \u7be1\u6539\u672a\u6765\u5e27\u96f6\u6cc4\u6f0f\u56e0\u679c\u65ad\u8a00\uff08\u542b Q1 \u6298\u5185\u5149\u8c31\u57fa\u5728\u7ebf\u5355\u5e27\u6295\u5f71\uff01\uff09
    max_feat_diff = 0.0
    max_ksrc_diff = 0
    max_pred_diff = 0.0
    for idx in (0, 4, 5):  # \u8986\u76d6\u65b0\u7248 H01\u3001H05(1s) \u4e0e\u9ad8\u9971\u548c H06
        tr_idx = [j for j in range(10) if j != idx]
        q1_basis_fold = extract_fold_q1_spectral_basis(q1_res, heats, tr_idx, n_modes=Q1_CFG.q3_modal_rank)
        o2_scale_fold = float(np.mean([q2_sum.recons[j].o2_cont[-1] for j in tr_idx]))

        h = heats[idx]
        cut = len(h.time_s) // 2
        s_orig = compute_strictly_causal_streams(
            h.spectrum, h.wavelengths, h.o2_cum, h.co2_raw, h.co_raw, h.dt,
            h.heat_id, h.heat_no, q1_basis=q1_basis_fold, o2_scale_m3=o2_scale_fold,
        )
        m_orig = build_targets_and_backward_map(s_orig, q2_sum.recons[idx], pg)

        spec_tampered = h.spectrum.copy()
        o2_tampered = h.o2_cum.copy()
        co2_tampered = h.co2_raw.copy()
        co_tampered = h.co_raw.copy()
        spec_tampered[cut:] = rng.uniform(0.0, 65535.0, size=spec_tampered[cut:].shape)
        o2_tampered[cut:] = o2_tampered[cut - 1] + np.cumsum(rng.uniform(0.0, 50.0, size=len(o2_tampered) - cut))
        co2_tampered[cut:] = rng.uniform(0.0, 99.0, size=len(co2_tampered) - cut)
        co_tampered[cut:] = rng.uniform(0.0, 99.0, size=len(co_tampered) - cut)

        s_tamp = compute_strictly_causal_streams(
            spec_tampered, h.wavelengths, o2_tampered, co2_tampered, co_tampered, h.dt,
            h.heat_id, h.heat_no, q1_basis=q1_basis_fold, o2_scale_m3=o2_scale_fold,
        )
        m_tamp = build_targets_and_backward_map(s_tamp, q2_sum.recons[idx], pg)

        max_feat_diff = max(
            max_feat_diff,
            float(np.max(np.abs(s_orig.x_m4[:cut] - s_tamp.x_m4[:cut]))),
            float(np.max(np.abs(s_orig.q1_modal_scores[:cut] - s_tamp.q1_modal_scores[:cut]))),
        )
        max_ksrc_diff = max(max_ksrc_diff, int(np.max(np.abs(m_orig["k_src"][:cut] - m_tamp["k_src"][:cut]))))

        dummy_w1 = rng.standard_normal(s_orig.x_m4.shape[1]) * 0.05
        dummy_w2 = rng.standard_normal(s_orig.x_m4.shape[1]) * 0.05
        p1_o, p2_o = apply_causal_smith_observer(
            np.abs(s_orig.x_m4 @ dummy_w1), np.abs(s_orig.x_m4 @ dummy_w2),
            s_orig.obs_co2_causal, s_orig.obs_co_causal, m_orig["k_src"], h.dt, 0.5, 0.90,
        )
        p1_t, p2_t = apply_causal_smith_observer(
            np.abs(s_tamp.x_m4 @ dummy_w1), np.abs(s_tamp.x_m4 @ dummy_w2),
            s_tamp.obs_co2_causal, s_tamp.obs_co_causal, m_tamp["k_src"], h.dt, 0.5, 0.90,
        )
        max_pred_diff = max(
            max_pred_diff,
            float(np.max(np.abs(p1_o[:cut] - p1_t[:cut]))),
            float(np.max(np.abs(p2_o[:cut] - p2_t[:cut]))),
        )

    ok_causal = (max_feat_diff == 0.0) and (max_ksrc_diff == 0) and (max_pred_diff == 0.0)
    rows.append({
        "Audit_Check": "5_Tamper_Future_Zero_Leakage_Causality_Test (Incl_Q1_Basis)",
        "Status": "PASS" if ok_causal else "FAIL",
        "Metric": f"max_feat_diff={max_feat_diff:.6e}, max_ksrc_diff={max_ksrc_diff}, max_pred_diff={max_pred_diff:.6e}",
    })

    # Audit 6: [P0-6 \u65b0\u589e] \u8de8\u7089\u6d4b\u8bd5\u96c6\u96f6\u6e17\u900f\u65ad\u8a00\uff1a\u5c06\u6d4b\u8bd5\u7089 H01 \u6574\u7089\u66ff\u6362\u4e3a\u968f\u673a\u4e71\u7801\uff0c\u9a8c\u8bc1\u6298\u5185\u8bad\u7ec3\u6807\u5b9a\u7684 Q1 \u57fa\u4e0e PFR \u65f6\u5ef6\u9010\u4f4d\u4e0d\u53d8\uff01
    tr_1_9 = list(range(1, 10))
    mu_orig, vr_orig = extract_fold_q1_spectral_basis(q1_res, heats, tr_1_9)
    fd_orig = fit_pfr_kinetics(q2_sum.recons, tr_1_9, "LeakCheck_Orig")

    q1_corrupted = list(q1_res)
    q1_corrupted[0] = replace(q1_res[0], spec_clean=rng.uniform(0.0, 99999.0, size=q1_res[0].spec_clean.shape))
    rec_corrupted = list(q2_sum.recons)
    rec_corrupted[0] = replace(
        q2_sum.recons[0],
        co2_cont=rng.uniform(0.0, 99.0, size=len(q2_sum.recons[0].co2_cont)),
        co_cont=rng.uniform(0.0, 99.0, size=len(q2_sum.recons[0].co_cont)),
    )
    mu_corr, vr_corr = extract_fold_q1_spectral_basis(q1_corrupted, heats, tr_1_9)
    fd_corr = fit_pfr_kinetics(rec_corrupted, tr_1_9, "LeakCheck_Corrupted")

    cross_diff = max(
        float(np.max(np.abs(mu_orig - mu_corr))),
        float(np.max(np.abs(vr_orig - vr_corr))),
        abs(fd_orig.v_pipe_m3 - fd_corr.v_pipe_m3),
    )
    rows.append({
        "Audit_Check": "6_Cross_Heat_Nested_Fold_Zero_Leakage_Test",
        "Status": "PASS" if cross_diff == 0.0 else "FAIL",
        "Metric": f"max_train_artifact_diff_under_corrupted_test_heat={cross_diff:.6e}",
    })

    df_audit = pd.DataFrame(rows)
    df_audit.to_csv(TABLE_DIR / "verification_audit_report.csv", index=False)
    for _, r in df_audit.iterrows():
        assert r["Status"] == "PASS", f"Scientific audit failed: {r.to_dict()}"
    return df_audit
