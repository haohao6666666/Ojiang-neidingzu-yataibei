"""Sensitivity of Problem 3 to a misestimated plug-flow volume dV_pipe (x0.9, x1.0, x1.1).
The system trains and corrects with the perturbed delay; the test heat is always scored against
targets built with the calibrated (true) delay. Runs inside a throw-away worktree of commit 416495a."""
import dataclasses, shutil, sys, time
from pathlib import Path
CODE = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE))
import config
from io_loader import load_all_heats
from q1_spectral_recon import run_question1
from q2_gas_delay import run_question2
import q3_fusion_predict as q3m

config.ensure_dirs(); config.setup_plot_style()
heats = load_all_heats(use_cache=True)
t0 = time.time(); q1 = run_question1(heats, make_plots=False); print("q1 done", round(time.time()-t0), flush=True)
t0 = time.time(); q2 = run_question2(heats, make_plots=False); print("q2 done", round(time.time()-t0), flush=True)

orig_fit, orig_build = q3m.fit_pfr_kinetics, q3m.build_targets_and_backward_map
for factor in (1.0, 0.9, 1.1):
    def fit_scaled(*a, **k):
        d = orig_fit(*a, **k)
        if d.scope.startswith("Fold_Excl"):
            return dataclasses.replace(d, v_pipe_m3=d.v_pipe_m3 * factor, scope=f"{d.scope}|nom={d.v_pipe_m3!r}")
        return d
    def build_wrap(stream, recon, delay):
        m = orig_build(stream, recon, delay)
        if "|nom=" in delay.scope and delay.scope.startswith(f"Fold_Excl_H{stream.heat_id:02d}|"):
            nom = float(delay.scope.split("|nom=")[1])
            mn = orig_build(stream, recon, dataclasses.replace(delay, v_pipe_m3=nom))
            m = dict(m); m["y_co2"], m["y_co"] = mn["y_co2"], mn["y_co"]
            m["valid"] = mn["valid"] & m["valid"]
        return m
    q3m.fit_pfr_kinetics, q3m.build_targets_and_backward_map = fit_scaled, build_wrap
    t0 = time.time()
    out = q3m.run_question3(heats, q1, q2, make_plots=False)
    out["summary"].to_csv(OUT / f"q3_summary_factor_{factor:.1f}.csv", index=False)
    print(f"factor {factor} done", round(time.time()-t0), flush=True)
q3m.fit_pfr_kinetics, q3m.build_targets_and_backward_map = orig_fit, orig_build
print("ALL DONE")
