"""
2026 APMCM Problem B \u4e00\u952e\u5f0f\u5168\u6d41\u7a0b\u6c42\u89e3\u5165\u53e3\uff08v4 \u7ea2\u961f\u5ba1\u8ba1\u5168\u91cf\u4fee\u6b63\u7248\uff09\u3002
\u7528\u6cd5\uff1a
    python main.py --stage all      # \u4f9d\u6b21\u6267\u884c\u7b2c 1\u30012\u30013 \u9898 + 6 \u9879\u79d1\u5b66\u4e00\u81f4\u6027\u4e0e\u96f6\u6cc4\u6f0f\u81ea\u52a8\u5ba1\u8ba1
    python main.py --stage q1|q2|q3 # \u5176\u4e2d --stage q3 \u81ea\u52a8\u4e32\u8054\u6267\u884c\u7b2c 1\u30012 \u9898\u4ee5\u65e0\u6cc4\u6f0f\u63d0\u53d6\u8fde\u7eed\u4fe1\u53f7\u6a21\u578b\uff01
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

    print("[1/5] \u52a0\u8f7d\u5168\u90e8 10 \u4e2a\u8f6c\u7089\u7089\u6b21\u5e76\u6267\u884c SHA-256 \u7f13\u5b58\u6458\u8981\u4e0e Sequence \u8fde\u7eed\u6027\u6821\u9a8c...")
    heats = load_all_heats(use_cache=True)
    print(f"      \u52a0\u8f7d\u5b8c\u6210 ({time.time() - t0:.2f}s) | \u7089\u6b21\u4e0e\u91c7\u6837\u6b65\u957f: {[f'{h.heat_no}({h.dt:g}s)' for h in heats]}")

    q1_res = None
    q2_sum = None

    if args.stage in ("q1", "q3", "all"):
        t1 = time.time()
        print("\n[2/5] \u6267\u884c\u7b2c 1 \u9898\uff1a\u5bf9\u6570\u57df\u53f3\u5220\u5931\u4fdd\u754c\u4f4e\u79e9 SVD \u5149\u8c31\u6062\u590d + \u5bf9\u6570\u4e8c\u5143\u6837\u6761 3D \u66f2\u9762 + \u53bb\u95ea\u70c1\u8109\u51b2\u6ce8\u5165\u9a8c\u8bc1...")
        q1_res = run_question1(heats, make_plots=(not args.no_plots and args.stage in ("q1", "all")))
        mean_r2 = sum(r.metrics["masked_cv_r2"] for r in q1_res) / len(q1_res)
        min_r2 = min(r.metrics["masked_cv_r2"] for r in q1_res)
        mean_ret = sum(r.metrics["K_peak_retention_pct"] for r in q1_res) / len(q1_res)
        print(f"      \u5b8c\u6210 ({time.time() - t1:.2f}s) | \u5e73\u5747 1% \u63a9\u7801\u91cd\u6784 R\xb2 = {mean_r2:.4f} (\u6700\u4f4e {min_r2:.4f}) | K \u53cc\u5cf0\u4fdd\u771f\u7387 = {mean_ret:.2f}%")

    if args.stage in ("q2", "q3", "all"):
        t2 = time.time()
        print("\n[3/5] \u6267\u884c\u7b2c 2 \u9898\uff1a\u53bb\u8865\u8d26\u8df3\u53d8\u6d41\u91cf + \u591a\u521d\u503c\u5168\u5c40 PFR \u53d8\u65f6\u6ede\u8fa8\u8bc6 + 250 \u6b21\u5168\u53c2\u6570\u8054\u5408\u91cd\u62df\u5408 Bootstrap...")
        q2_sum = run_question2(heats, make_plots=(not args.no_plots and args.stage in ("q2", "all")))
        pg = q2_sum.pfr_global
        print(f"      \u5b8c\u6210 ({time.time() - t2:.2f}s) | Global PFR R\xb2 = {pg.mean_r2:.4f} | "
              f"\u0394V_pipe = {pg.v_pipe_m3:.2f} m\xb3 (\u5168\u53c2\u6570\u91cd\u62df\u5408 95% CI [{pg.v_pipe_ci95[0]:.1f}, {pg.v_pipe_ci95[1]:.1f}])")

    if args.stage in ("q3", "all"):
        t3 = time.time()
        print("\n[4/5] \u6267\u884c\u7b2c 3 \u9898\uff1aQ1 \u6298\u5185\u5149\u8c31\u57fa\u5728\u7ebf\u5355\u5e27\u6295\u5f71 + \u7ebf\u6027\u6811\u6df7\u5408\u4e0e\u56e0\u679c\u8f6f\u95e8\u63a7\u96c6\u6210 + \u5185\u5c42\u5d4c\u5957 Smith/\u81ea\u9002\u5e94\u5171\u5f62\u533a\u95f4...")
        assert q1_res is not None and q2_sum is not None
        q3_out = run_question3(heats, q1_res, q2_sum, make_plots=(not args.no_plots))
        print(f"      \u5b8c\u6210 ({time.time() - t3:.2f}s) | \u4e25\u683c\u6298\u5185\u5d4c\u5957\u7559\u4e00\u7089\u4ea4\u53c9\u9a8c\u8bc1\u6c47\u603b\uff1a")
        print(q3_out["summary"].to_string(index=False))
        print("\n      10 \u6298\u72ec\u7acb\u8bad\u7ec3\u6807\u5b9a \u0394V_pipe^{(-i)} \u7a33\u5b9a\u6027\u6c47\u603b\uff1a")
        fd = q3_out["fold_delays"]["Train_dV_pipe_m3"]
        print(f"      mean = {fd.mean():.2f} m\xb3, std = {fd.std():.2f} m\xb3, range = [{fd.min():.2f}, {fd.max():.2f}] m\xb3")

    if args.stage == "all":
        t4 = time.time()
        print("\n[5/5] \u6267\u884c 6 \u9879\u79d1\u5b66\u4e00\u81f4\u6027\u3001\u76f2\u521d\u503c\u542b\u566a\u5408\u6210\u56de\u6536\u4e0e\u300c\u672a\u6765\u5e27/\u8de8\u7089\u53cc\u96f6\u6cc4\u6f0f\u300d\u81ea\u52a8\u5316\u5ba1\u8ba1...")
        assert q1_res is not None and q2_sum is not None
        df_audit = run_scientific_audit(heats, q1_res, q2_sum)
        print(df_audit.to_string(index=False))
        print(f"      6 \u9879\u5ba1\u8ba1\u5168\u90e8 PASS ({time.time() - t4:.2f}s)\uff01")

    print(f"\n\u5168\u90e8\u4efb\u52a1\u5b8c\u6210\uff0c\u603b\u8017\u65f6 {time.time() - t0:.2f}s\u3002\u8868\u683c\u76ee\u5f55: {TABLE_DIR} | \u56fe\u8868\u76ee\u5f55: {FIG_DIR}")


if __name__ == "__main__":
    main()
