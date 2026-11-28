"""
\u9ad8\u901f\u53cc\u6a21\u6001\u6570\u636e\u52a0\u8f7d\u5668\uff08v4 \u7ea2\u961f\u5ba1\u8ba1\u4fee\u6b63\u7248\uff09\uff1a
- Rust \u5f15\u64ce `python-calamine` \u4f18\u5148\uff08\u81ea\u52a8\u964d\u7ea7 `pyxlsb`\uff09
- \u6309\u5217\u4f4d\u7f6e\u517c\u5bb9\u4e2d\u82f1\u6587\u8868\u5934\uff1b\u65b0\u7248\u9644\u4ef6 1\uff086B03205\uff09\u6b65\u957f\u4e3a 0.5 s\uff0c\u4ec5\u9644\u4ef6 5\uff086C02057\uff09\u5b9e\u8bc1\u6b65\u957f\u4e3a 1.0 s
- \u663e\u5f0f\u8bfb\u53d6\u5e76\u4e25\u683c\u6821\u9a8c\u7b2c 0 \u5217 Sequence \u8fde\u7eed\u6027\uff081..T \u65e0\u8df3\u53f7/\u65ad\u5e27\u6f0f\u68c0\uff09
- \u91c7\u7528\u6587\u4ef6\u5185\u5bb9 SHA-256 \u6458\u8981\u4f5c\u4e3a `.npz` \u7f13\u5b58\u952e\uff0c\u5f7b\u5e95\u675c\u7edd\u540c\u4f53\u79ef\u6587\u4ef6\u66ff\u6362\u5bfc\u81f4\u7684\u9648\u65e7\u7f13\u5b58\u547d\u4e2d
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from config import CACHE_DIR, DATA_DIR, HEAT_DT, ensure_dirs


@dataclass
class HeatDataset:
    heat_id: int                 # 1..10
    heat_no: str                 # '6B03205' .. '6C02108'
    converter: str               # '6B' \u6216 '6C'
    dt: float                    # \u91c7\u6837\u95f4\u9694 (s)\uff1a\u9644\u4ef6 5 \u4e3a 1.0 s\uff0c\u5176\u4f59 9 \u7089\u5747\u4e3a 0.5 s
    sequence: np.ndarray         # (T,) \u539f\u59cb\u5e8f\u5217\u53f7\u5217\uff08\u5df2\u6821\u9a8c\u4e25\u683c\u7b49\u4e8e 1..T\uff09
    time_s: np.ndarray           # (T,) \u65f6\u95f4\u8f74 (s)
    wavelengths: np.ndarray      # (L,) \u6821\u51c6\u6ce2\u957f\u8f74 (nm), L=2047
    spectrum: np.ndarray         # (T, L) \u539f\u59cb\u706b\u7130\u5149\u8c31\u77e9\u9635
    o2_cum: np.ndarray           # (T,) \u7d2f\u79ef\u8017\u6c27\u91cf (m\xb3)
    co2_raw: np.ndarray          # (T,) CO2 \u6d4b\u91cf\u5e8f\u5217
    co_raw: np.ndarray           # (T,) CO \u6d4b\u91cf\u5e8f\u5217
    sha256_prefix: str = ""      # \u539f\u59cb\u6570\u636e\u6587\u4ef6 SHA-256 \u524d 16 \u4f4d\u6458\u8981


def _file_sha256_prefix(path: Path, length: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def _read_xlsb_rows(path: Path):
    try:
        from python_calamine import CalamineWorkbook
        wb = CalamineWorkbook.from_path(str(path))
        return wb.sheet_names[0], wb.get_sheet_by_index(0).to_python()
    except ImportError:
        import pyxlsb
        with pyxlsb.open_workbook(str(path)) as wb:
            sheet_name = wb.sheets[0]
            with wb.get_sheet(1) as sheet:
                rows = [[cell.v for cell in row] for row in sheet.rows()]
        return sheet_name, rows


def _extract_attachment_index(path: Path) -> int:
    m = re.search(r"Attachment\s*(\d+)|\u9644\u4ef6\s*(\d+)", path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse attachment index from filename: {path.name}")
    return int(m.group(1) or m.group(2))


def _validate_sequence_column(seq: np.ndarray, path_name: str) -> np.ndarray:
    """\u4e25\u683c\u6821\u9a8c\u7b2c 0 \u5217\u5e8f\u5217\u53f7\u662f\u5426\u4e3a\u4ece 1 \u5f00\u59cb\u65e0\u8df3\u53f7\u3001\u65e0\u5012\u9000\u7684\u8fde\u7eed\u6574\u6570\u5e8f\u5217\u3002"""
    n_steps = len(seq)
    expected = np.arange(1, n_steps + 1, dtype=np.float64)
    if not np.allclose(seq, expected, atol=1e-6):
        diffs = np.diff(seq)
        bad_pos = np.where(np.abs(diffs - 1.0) > 1e-6)[0]
        raise ValueError(
            f"Sequence column discontinuity detected in {path_name} at indices {bad_pos[:10].tolist()}"
        )
    return seq.astype(np.int64)


def load_single_heat(path: Path, use_cache: bool = True, dt_override: float | None = None) -> HeatDataset:
    ensure_dirs()
    heat_id = _extract_attachment_index(path)
    dt = HEAT_DT[heat_id] if dt_override is None else dt_override
    sha_prefix = _file_sha256_prefix(path, length=16)
    cache_file = CACHE_DIR / f"heat_{heat_id:02d}_{sha_prefix}.npz"

    if use_cache and cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        if "sequence" in z:
            heat_no = str(z["heat_no"])
            seq = _validate_sequence_column(z["sequence"].astype(np.float64), path.name)
            n_steps = len(seq)
            return HeatDataset(
                heat_id=heat_id,
                heat_no=heat_no,
                converter=heat_no[:2],
                dt=dt,
                sequence=seq,
                time_s=(seq - 1).astype(np.float64) * dt,
                wavelengths=z["wavelengths"].astype(np.float64),
                spectrum=z["spectrum"].astype(np.float64),
                o2_cum=z["o2_cum"].astype(np.float64),
                co2_raw=z["co2_raw"].astype(np.float64),
                co_raw=z["co_raw"].astype(np.float64),
                sha256_prefix=sha_prefix,
            )

    sheet_name, rows = _read_xlsb_rows(path)
    wavelengths = np.array([float(v) for v in rows[0][1:-3]], dtype=np.float64)
    arr = np.asarray(rows[1:], dtype=np.float64)
    seq = _validate_sequence_column(arr[:, 0], path.name)
    heat_no = sheet_name.strip()
    time_s = (seq - 1).astype(np.float64) * dt

    if use_cache:
        # \u6e05\u7406\u540c\u7089\u53f7\u5386\u53f2\u65e7\u7f13\u5b58\u6587\u4ef6\uff0c\u907f\u514d\u50f5\u5c38\u7f13\u5b58\u5806\u79ef
        for old_f in CACHE_DIR.glob(f"heat_{heat_id:02d}_*.npz"):
            if old_f != cache_file:
                old_f.unlink(missing_ok=True)
        np.savez_compressed(
            cache_file,
            heat_no=np.array(heat_no),
            sequence=seq,
            wavelengths=wavelengths,
            spectrum=arr[:, 1:-3].astype(np.float32),
            o2_cum=arr[:, -3],
            co2_raw=arr[:, -2],
            co_raw=arr[:, -1],
        )

    return HeatDataset(
        heat_id=heat_id,
        heat_no=heat_no,
        converter=heat_no[:2],
        dt=dt,
        sequence=seq,
        time_s=time_s,
        wavelengths=wavelengths,
        spectrum=arr[:, 1:-3],
        o2_cum=arr[:, -3],
        co2_raw=arr[:, -2],
        co_raw=arr[:, -1],
        sha256_prefix=sha_prefix,
    )


def load_all_heats(data_dir: Path = DATA_DIR, use_cache: bool = True) -> List[HeatDataset]:
    files = sorted(data_dir.glob("*.xlsb"), key=_extract_attachment_index)
    if len(files) != 10:
        raise FileNotFoundError(f"Expected 10 xlsb files in {data_dir}, found {len(files)}.")
    return [load_single_heat(f, use_cache=use_cache) for f in files]
