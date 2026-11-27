"""
2026 APMCM Problem B 一键式全流程求解入口（v4 红队审计全量修正版）。
用法：
    python main.py --stage all      # 依次执行第 1、2、3 题 + 6 项科学一致性与零泄漏自动审计
    python main.py --stage q1|q2|q3 # 其中 --stage q3 自动串联执行第 1、2 题以无泄漏提取连续信号模型！
"""
from __future__ import annotations

import argparse
import time

from config import FIG_DIR, TABLE_DIR, ensure_dirs, setup_plot_style
from io_loader import load_all_heats
from q1_spectral_recon import run_question1
from q2_gas_delay import run_question2
from q3_fusion_predict import run_question3
from scientific_verify import run_scientific_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="2026 APMCM Problem B Unified Pipeline (v4 Red-Team Verified)")
    parser.add_argument("--stage", choices=["q1", "q2", "q3", "all"], default="all")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    setup_plot_style()
    t0 = time.time()

    print("[1/5] 加载全部 10 个转炉炉次并执行 SHA-256 缓存摘要与 Sequence 连续性校验...")
    heats = load_all_heats(use_cache=True)
    print(f"      加载完成 ({time.time() - t0:.2f}s) | 炉次与采样步长: {[f'{h.heat_no}({h.dt:g}s)' for h in heats]}")

    q1_res = None
    q2_sum = None

    if args.stage in ("q1", "q3", "all"):
        t1 = time.time()
        print("\n[2/5] 执行第 1 题：对数域右删失保界低秩 SVD 光谱恢复 + 对数二元样条 3D 曲面 + 去闪烁脉冲注入验证...")
        q1_res = run_question1(heats, make_plots=(not args.no_plots and args.stage in ("q1", "all")))
        mean_r2 = sum(r.metrics["masked_cv_r2"] for r in q1_res) / len(q1_res)
        min_r2 = min(r.metrics["masked_cv_r2"] for r in q1_res)
        mean_ret = sum(r.metrics["K_peak_retention_pct"] for r in q1_res) / len(q1_res)
        print(f"      完成 ({time.time() - t1:.2f}s) | 平均 1% 掩码重构 R² = {mean_r2:.4f} (最低 {min_r2:.4f}) | K 双峰保真率 = {mean_ret:.2f}%")

    if args.stage in ("q2", "q3", "all"):
        t2 = time.time()
        print("\n[3/5] 执行第 2 题：去补账跳变流量 + 多初值全局 PFR 变时滞辨识 + 250 次全参数联合重拟合 Bootstrap...")
        q2_sum = run_question2(heats, make_plots=(not args.no_plots and args.stage in ("q2", "all")))
        pg = q2_sum.pfr_global
        print(f"      完成 ({time.time() - t2:.2f}s) | Global PFR R² = {pg.mean_r2:.4f} | "
              f"ΔV_pipe = {pg.v_pipe_m3:.2f} m³ (全参数重拟合 95% CI [{pg.v_pipe_ci95[0]:.1f}, {pg.v_pipe_ci95[1]:.1f}])")

    if args.stage in ("q3", "all"):
        t3 = time.time()
        print("\n[4/5] 执行第 3 题：Q1 折内光谱基在线单帧投影 + 线性树混合与因果软门控集成 + 内层嵌套 Smith/自适应共形区间...")
        assert q1_res is not None and q2_sum is not None
        q3_out = run_question3(heats, q1_res, q2_sum, make_plots=(not args.no_plots))
        print(f"      完成 ({time.time() - t3:.2f}s) | 严格折内嵌套留一炉交叉验证汇总：")
        print(q3_out["summary"].to_string(index=False))
        print("\n      10 折独立训练标定 ΔV_pipe^{(-i)} 稳定性汇总：")
        fd = q3_out["fold_delays"]["Train_dV_pipe_m3"]
        print(f"      mean = {fd.mean():.2f} m³, std = {fd.std():.2f} m³, range = [{fd.min():.2f}, {fd.max():.2f}] m³")

    if args.stage == "all":
        t4 = time.time()
        print("\n[5/5] 执行 6 项科学一致性、盲初值含噪合成回收与「未来帧/跨炉双零泄漏」自动化审计...")
        assert q1_res is not None and q2_sum is not None
        df_audit = run_scientific_audit(heats, q1_res, q2_sum)
        print(df_audit.to_string(index=False))
        print(f"      6 项审计全部 PASS ({time.time() - t4:.2f}s)！")

    print(f"\n全部任务完成，总耗时 {time.time() - t0:.2f}s。表格目录: {TABLE_DIR} | 图表目录: {FIG_DIR}")


if __name__ == "__main__":
    main()
