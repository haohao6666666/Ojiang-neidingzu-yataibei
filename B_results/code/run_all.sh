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
