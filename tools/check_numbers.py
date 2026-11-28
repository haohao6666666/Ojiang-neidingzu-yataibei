#!/usr/bin/env python3
"""Check that language polishing did not change any number in the manuscript.

    python3 tools/check_numbers.py                     # working file vs last commit
    python3 tools/check_numbers.py --base 3cbf1c9      # working file vs a given commit
    python3 tools/check_numbers.py --file other.md

Every number (including decimals, ranges, scientific notation and equation tags) is
extracted from the base version and the current version. Numbers that appear a different
number of times are listed. A rewritten sentence can legitimately drop a repeated number,
so judge each hit in context; a changed value (e.g. 0.873 -> 0.87) is never allowed.
"""
import argparse
import re
import subprocess
import sys
from collections import Counter

NUM = re.compile(r"(?<![A-Za-z_])\d+(?:[.,]\d+)*(?:\s*\\times\s*10\^\{?-?\d+\}?)?")


def numbers(text):
    return Counter(m.group(0).replace(" ", "") for m in NUM.finditer(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="B题论文文稿.md")
    ap.add_argument("--base", default="HEAD", help="git revision to compare against")
    a = ap.parse_args()

    try:
        old = subprocess.run(["git", "show", f"{a.base}:{a.file}"], capture_output=True,
                             check=True).stdout.decode("utf-8")
    except subprocess.CalledProcessError as e:
        sys.exit(f"cannot read {a.file} at {a.base}: {e.stderr.decode().strip()}")
    new = open(a.file, encoding="utf-8").read()

    c_old, c_new = numbers(old), numbers(new)
    lost = {k: c_old[k] - c_new[k] for k in c_old if c_old[k] > c_new[k]}
    added = {k: c_new[k] - c_old[k] for k in c_new if c_new[k] > c_old[k]}

    if not lost and not added:
        print(f"OK: all {sum(c_old.values())} numbers unchanged against {a.base}.")
        return
    if lost:
        print("Numbers that appear fewer times now (check they were not changed or dropped by mistake):")
        for k, v in sorted(lost.items()):
            print(f"  {k}  (-{v})")
    if added:
        print("Numbers that appear more times now (each must come from a result table):")
        for k, v in sorted(added.items()):
            print(f"  {k}  (+{v})")
    sys.exit(1)


if __name__ == "__main__":
    main()
