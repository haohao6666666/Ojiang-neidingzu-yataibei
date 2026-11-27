"""Step 0: extract Data.7z, convert .xlsb -> .csv (LibreOffice), cache each heat as npz.

Attachment 1 uses the corrected file (flue-gas columns filled); the copy inside Data.7z
has all-zero CO/CO2 and is NOT used.
Usage: python q0_prepare.py  (requires bsdtar and soffice on PATH)
"""
import glob
import os
import re
import shutil
import subprocess
import numpy as np
import pandas as pd
from common import RAW

PROB = os.path.join(os.path.dirname(os.path.dirname(RAW)), "2026 APMCM Problem B", "Data.7z")
XLSB = os.path.join(RAW, "xlsb")
CSV = os.path.join(RAW, "csv")
os.makedirs(CSV, exist_ok=True)

# 1) extract attachments 2-10 from the official archive (skipped if all ten .xlsb files are already present,
#    e.g. in the GitHub copy, which ships the ten .xlsb files and not Data.7z)
os.makedirs(XLSB, exist_ok=True)
if len(glob.glob(os.path.join(XLSB, "*.xlsb"))) < 10:
    tmp = os.path.join(RAW, "_extract")
    os.makedirs(tmp, exist_ok=True)
    subprocess.run(["tar", "-xf", PROB, "-C", tmp], check=True)
    for f in glob.glob(os.path.join(tmp, "Data", "*.xlsb")):
        k = int(re.search(r"Attachment (\d+)", f).group(1))
        if k == 1:
            continue  # replaced by corrected file
        os.replace(f, os.path.join(XLSB, os.path.basename(f)))
    shutil.rmtree(tmp)          # the old attachment 1 (all-zero gas) stays out of the pipeline

# 2) convert to csv
files = sorted(glob.glob(os.path.join(XLSB, "*.xlsb")))
subprocess.run(["soffice", "--headless", "--convert-to", "csv", "--outdir", CSV] + files,
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 3) cache as npz (columns taken by position: seq | 2047 spectra | O2 | CO2 | CO)
for f in sorted(glob.glob(os.path.join(CSV, "*.csv"))):
    k = int(re.search(r"Attachment (\d+)", f).group(1))
    df = pd.read_csv(f)
    wl = pd.to_numeric(pd.Series(df.columns[1:-3]), errors="coerce").values.astype(float)
    df = df.apply(pd.to_numeric, errors="coerce")
    spec = df.iloc[:, 1:-3].values.astype(float)
    o2, co2, co = (df.iloc[:, -3].values.astype(float), df.iloc[:, -2].values.astype(float),
                   df.iloc[:, -1].values.astype(float))
    assert spec.shape[1] == 2047 and not np.isnan(spec).any()
    np.savez_compressed(os.path.join(RAW, f"heat{k:02d}.npz"), wl=wl, spec=spec, o2=o2, co2=co2, co=co)
    print(f"heat {k:2d}: n={len(o2)}, wl {wl[0]:.3f}-{wl[-1]:.3f}, CO max={co.max():.2f}, CO2 max={co2.max():.2f}")
