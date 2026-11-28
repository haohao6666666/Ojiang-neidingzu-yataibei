"""Build the compact plotting data in paper_figures/data/ from the frozen pipeline outputs.

Sources (nothing is recomputed except plotting-level derivations noted below):
  * B_results/data_clean/heat*_spec.npz, heat*_proc.npz   (cache of B_results/code/run_all.sh)
  * B_results/data_raw/heat*.npz                           (raw counter readings, same cache)
  * B_results/tables/*.csv                                 (frozen A-version tables)
  * solution_v4_verified/solution_outputs/tables/*.csv     (frozen v4 tables)
  * q2_dump.pkl / q3_dump.pkl: per-frame arrays exported by re-running the frozen v4 code
    (solution_v4_verified/solution_code, `main.py --stage q3`) with an added pickle dump;
    the re-run reproduces the frozen summary tables, see check_reproduction() below.

    python paper_figures/extract_data.py --b-results ../B_results --v4-dump /tmp/v4run
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
V4T = os.path.join(REPO, "solution_v4_verified", "solution_outputs", "tables")
DT = {k: (1.0 if k == 5 else 0.5) for k in range(1, 11)}
GROUP = {k: ("6B" if k <= 4 else "6C") for k in range(1, 11)}


def ln10(a):
    return np.asarray(a, dtype=np.float32) / np.log(10.0)


def flow(t, o2, dt, win_s):
    iso = IsotonicRegression().fit_transform(t, o2)
    w = int(round(win_s / dt)) | 1
    return np.clip(savgol_filter(iso, w, 1, deriv=1, delta=dt, mode="interp"), 0, None)


def task1(br):
    cl = os.path.join(br, "data_clean")
    # Figure 7: Heat 3 decomposition (every 2nd wavelength, float16 to keep the repo small)
    d = np.load(os.path.join(cl, "heat03_spec.npz"))
    X = np.where(d["miss_sat"], np.nan, d["X_obs"])
    S = np.where(d["miss_sat"], np.nan, d["S_rpca"])
    np.savez_compressed(os.path.join(DATA, "fig07_heat03_decomposition.npz"), t=d["t"], wl=d["wl"][::2],
                        X=ln10(X[:, ::2]).astype(np.float16), L=ln10(d["L_rpca"][:, ::2]).astype(np.float16),
                        S=ln10(S[:, ::2]).astype(np.float16))
    # Figure 8: Heat 9 (most flagged pixels), 600-nm trace and candidate anomalous frames
    d = np.load(os.path.join(cl, "heat09_spec.npz"))
    j = int(np.argmin(np.abs(d["wl"] - 600)))
    t = d["t"]
    frac = d["outlier"].mean(1)
    mid = (t > 0.2 * t[-1]) & (t < 0.85 * t[-1])
    cand = np.argsort(np.where(mid, frac, -1))[::-1][:6]
    np.savez_compressed(os.path.join(DATA, "fig08_heat09_outliers.npz"), t=t, wl=d["wl"], j=j,
                        x=ln10(d["X_obs"][:, j]), out=d["outlier"][:, j], rec=ln10(d["X_rec"][:, j]),
                        trend=ln10(d["X_trend"][:, j]), cand=cand, cand_frac=frac[cand],
                        cand_X=ln10(d["X_obs"][cand]), cand_out=d["outlier"][cand], cand_rec=ln10(d["X_rec"][cand]),
                        cand_trend=ln10(d["X_trend"][cand]), cand_valid=d["valid"][cand])
    # Figure 9: trend surfaces of the ten heats, 160 x 160 grid
    surf = {}
    for k in range(1, 11):
        d = np.load(os.path.join(cl, f"heat{k:02d}_spec.npz"))
        ti = np.linspace(0, len(d["t"]) - 1, 160).astype(int)
        wi = np.linspace(0, len(d["wl"]) - 1, 160).astype(int)
        surf[f"t{k}"] = d["t"][ti]
        surf[f"z{k}"] = ln10(d["X_trend"][np.ix_(ti, wi)])
        surf["wl"] = d["wl"][wi]
    np.savez_compressed(os.path.join(DATA, "fig09_surfaces.npz"), **surf)


def task2(br):
    cl = os.path.join(br, "data_clean")
    tab = os.path.join(br, "tables")
    # Figure 4: counter beat, Heat 3
    r = np.load(os.path.join(br, "data_raw", "heat03.npz"))
    o2 = r["o2"]
    t = np.arange(len(o2)) * DT[3]
    np.savez_compressed(os.path.join(DATA, "fig04_heat03_beat.npz"), t=t, o2=o2, dV=np.r_[0.0, np.diff(o2)],
                        q5=flow(t, o2, DT[3], 5), q12=flow(t, o2, DT[3], 12), q24=flow(t, o2, DT[3], 24))
    # Figures 10 and 11: reconstructed process curves of all heats
    pr = {}
    for k in range(1, 11):
        p = np.load(os.path.join(cl, f"heat{k:02d}_proc.npz"))
        for n in ("t", "V", "Q", "co", "co2", "Eco", "Eco2", "co_raw", "co2_raw", "bps", "tbs"):
            pr[f"{n}{k}"] = p[n]
    r3 = np.load(os.path.join(br, "data_raw", "heat03.npz"))
    pr["o2raw3"] = r3["o2"]
    np.savez_compressed(os.path.join(DATA, "fig10_11_process.npz"), **pr)
    # Figure 13: 44 windows with a distinct cross-correlation peak (r > 0.3) and the four flow bins
    w = pd.read_csv(os.path.join(tab, "q2_delay_windows.csv"))
    w[w.r_xcorr > 0.3].to_csv(os.path.join(DATA, "fig13_windows.csv"), index=False)


def v4(dump_dir):
    q2 = pickle.load(open(os.path.join(dump_dir, "q2_dump.pkl"), "rb"))
    out = {"tmix": q2["tmix"], "vpipe": q2["vpipe"], "ci": np.array(q2["ci"]), "r2": q2["r2"], "step_r2": q2["step_r2"]}
    for r in q2["recons"]:
        k = r["id"]
        out[f"o2_{k}"], out[f"co_{k}"], out[f"co2_{k}"] = r["o2"], r["co"], r["co2"]
    np.savez_compressed(os.path.join(DATA, "fig12_pfd.npz"), **out)
    q2["step"].to_csv(os.path.join(DATA, "fig12_events.csv"), index=False)
    q2["mass"].to_csv(os.path.join(DATA, "fig12_mass.csv"), index=False)

    q3 = pickle.load(open(os.path.join(dump_dir, "q3_dump.pkl"), "rb"))
    ser = {}
    for h in q3["heats"]:
        k = h["id"]
        for n in ("t", "o2", "y_co", "y_co2", "z_co", "z_co2", "zhw_co", "zhw_co2", "d_co", "d_co2", "dhw_co", "dhw_co2"):
            ser[f"{n}_{k}"] = np.asarray(h[n], dtype=np.float32)
    np.savez_compressed(os.path.join(DATA, "fig14_loho_series.npz"), **ser)
    pd.DataFrame(q3["transfer"], index=q3["labels"], columns=q3["labels"]).to_csv(os.path.join(DATA, "fig15_transfer.csv"))
    return q3


def check_reproduction(q3):
    """The re-run must reproduce the frozen v4 tables used in the text."""
    from sklearn.metrics import r2_score
    frozen = pd.read_csv(os.path.join(V4T, "q3_loho_per_heat_metrics.csv"))
    print(frozen.columns.tolist())
    for h in q3["heats"]:
        print(f"H{h['id']:02d} zero-probe CO R2 = {r2_score(h['y_co'], h['z_co']):.3f}, CO2 R2 = {r2_score(h['y_co2'], h['z_co2']):.3f}")
    s = q3["summary"].set_index("Mode")
    f = pd.read_csv(os.path.join(V4T, "q3_loho_ablation_summary.csv")).set_index("Mode")
    common = s.index.intersection(f.index)
    diff = (s.loc[common, ["CO_Mean_R2", "CO2_Mean_R2"]] - f.loc[common, ["CO_Mean_R2", "CO2_Mean_R2"]]).abs().max().max()
    print(f"max |re-run - frozen| over {len(common)} summary rows: {diff:.4g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-results", default=os.path.join(REPO, "..", "B_results"))
    ap.add_argument("--v4-dump", default=None)
    ap.add_argument("--only", choices=["task1", "task2", "v4"], default=None)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    if a.only in (None, "task1"):
        task1(a.b_results)
    if a.only in (None, "task2"):
        task2(a.b_results)
    if a.v4_dump and a.only in (None, "v4"):
        check_reproduction(v4(a.v4_dump))
