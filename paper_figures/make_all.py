"""Re-draw every paper figure into paper_figures/out/ (PDF for LaTeX + 300-dpi PNG).

    python paper_figures/make_all.py

The compact plotting data in paper_figures/data/ is built once by extract_data.py.
"""
import glob
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
for script in sorted(glob.glob(os.path.join(here, "fig[0-9]*.py"))):
    print("->", os.path.basename(script))
    subprocess.run([sys.executable, script], cwd=here, check=True, stdout=subprocess.DEVNULL)
print("done:", len(glob.glob(os.path.join(here, "out", "*.pdf"))), "PDF files in", os.path.join(here, "out"))
