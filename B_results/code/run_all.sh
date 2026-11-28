#!/bin/bash
# Reproduce all results of APMCM 2026 Problem B (about 25 minutes on a laptop).
# Requires python3 (numpy, scipy, pandas, scikit-learn, lightgbm, matplotlib), bsdtar and LibreOffice (soffice).
set -e
cd "$(dirname "$0")"
python3 q0_prepare.py      # extract + convert raw data (uses corrected attachment 1)
python3 q1_spectra.py      # Problem 1
python3 q2a_process.py     # Problem 2: reconstruction, stages, relations, mass balance
python3 q2b_delay.py       # Problem 2: delay model (needs Q1 and Q2a outputs)
python3 q3_predict.py      # Problem 3 (needs Q1, Q2a, Q2b outputs)
python3 fig_q2_o2_beat.py  # Figure: O2 counter beat artefact (needs q0 and q2a outputs)
# Delay misestimation sensitivity (Table 9) runs on the frozen v4 code, about 40 minutes.
# Run it in a separate worktree of commit 416495a, because v4 rewrites its own output folder:
#   git worktree add /tmp/v4 416495a
#   python3 sens_delay_misestimation.py /tmp/v4/solution_v4_verified/solution_code <out_dir>
