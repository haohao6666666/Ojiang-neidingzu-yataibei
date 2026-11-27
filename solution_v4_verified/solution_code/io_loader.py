"""
高速双模态数据加载器（v4 红队审计修正版）：
- Rust 引擎 `python-calamine` 优先（自动降级 `pyxlsb`）
- 按列位置兼容中英文表头；新版附件 1（6B03205）步长为 0.5 s，仅附件 5（6C02057）实证步长为 1.0 s
- 显式读取并严格校验第 0 列 Sequence 连续性（1..T 无跳号/断帧漏检）
- 采用文件内容 SHA-256 摘要作为 `.npz` 缓存键，彻底杜绝同体积文件替换导致的陈旧缓存命中
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
    converter: str               # '6B' 或 '6C'
    dt: float                    # 采样间隔 (s)：附件 5 为 1.0 s，其余 9 炉均为 0.5 s
    sequence: np.ndarray         # (T,) 原始序列号列（已校验严格等于 1..T）
    time_s: np.ndarray           # (T,) 时间轴 (s)
    wavelengths: np.ndarray      # (L,) 校准波长轴 (nm), L=2047
    spectrum: np.ndarray         # (T, L) 原始火焰光谱矩阵
    o2_cum: np.ndarray           # (T,) 累积耗氧量 (m³)
    co2_raw: np.ndarray          # (T,) CO2 测量序列
    co_raw: np.ndarray           # (T,) CO 测量序列
    sha256_prefix: str = ""      # 原始数据文件 SHA-256 前 16 位摘要


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
    m = re.search(r"Attachment\s*(\d+)|附件\s*(\d+)", path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse attachment index from filename: {path.name}")
    return int(m.group(1) or m.group(2))


def _validate_sequence_column(seq: np.ndarray, path_name: str) -> np.ndarray:
    """严格校验第 0 列序列号是否为从 1 开始无跳号、无倒退的连续整数序列。"""
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
        # 清理同炉号历史旧缓存文件，避免僵尸缓存堆积
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
