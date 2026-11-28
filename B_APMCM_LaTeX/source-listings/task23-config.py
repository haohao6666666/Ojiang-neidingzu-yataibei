"""
\u5168\u5c40\u914d\u7f6e\u6a21\u5757\uff08v4 \u7ea2\u961f\u5ba1\u8ba1\u5168\u91cf\u4fee\u6b63\u7248\uff09\uff1a
- \u4e25\u683c\u533a\u5206\u5404\u7089\u5b9e\u8bc1\u91c7\u6837\u6b65\u957f\uff1a\u66f4\u65b0\u7248\u9644\u4ef6 1\uff086B03205\uff09\u53ca\u5176\u4f59 8 \u7089\u5747\u4e3a dt = 0.5 s\uff1b
  \u552f\u72ec\u9644\u4ef6 5\uff086C02057\uff0c775 \u5e27\uff09\u5b9e\u8bc1\u91c7\u6837\u6b65\u957f\u4e3a dt = 1.0 s\uff08\u9644 dt = 0.5 s \u654f\u611f\u6027\u5bf9\u7167\uff09\u3002
- \u79fb\u9664\u5168\u5c40 warnings.filterwarnings("ignore")\uff0c\u4ec5\u9488\u5bf9 Apple Silicon Accelerate AMX \u77e9\u9635\u534f\u5904\u7406\u5668
  \u5728\u6b63\u5e38\u6709\u9650\u6d6e\u70b9\u77e9\u9635\u4e58\u6cd5\u4e2d\u8bef\u7f6e IEEE \u7c98\u6ede\u5f02\u5e38\u4f4d\u7684\u5e95\u5c42\u5df2\u77e5\u95ee\u9898\u914d\u7f6e np.seterr\uff0c\u4fdd\u7559\u5168\u90e8\u771f\u5b9e\u8fd0\u884c\u544a\u8b66\u3002
- \u5168\u82f1\u6587 300 DPI PNG + \u77e2\u91cf PDF \u53cc\u5bfc\u51fa\uff0c\u76f4\u63a5\u9002\u914d APMCM LaTeX \u6392\u7248\u3002
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import matplotlib as mpl
import numpy as np

# \u4ec5\u5c4f\u853d Apple Silicon Accelerate BLAS dgemm \u7a7a\u95f2\u901a\u9053\u7c98\u6ede\u6d6e\u70b9\u6807\u5fd7\u4f4d\uff1b\u4fdd\u7559\u5168\u90e8 Python/SciPy/Sklearn warnings
np.seterr(divide="ignore", over="ignore", invalid="ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent


def _resolve_data_dir() -> Path:
    """\u81ea\u52a8\u517c\u5bb9\u672c\u5730\u8d5b\u9898\u5de5\u4f5c\u533a\u8def\u5f84\u4e0e GitHub \u4ed3\u5e93 B_results/data_raw/xlsb \u8def\u5f84\u3002"""
    candidates = [
        ROOT_DIR / "\u9644\u4ef6_Data" / "Data",
        ROOT_DIR.parent / "B_results" / "data_raw" / "xlsb",
        ROOT_DIR / "B_results" / "data_raw" / "xlsb",
    ]
    for cand in candidates:
        if cand.exists() and len(list(cand.glob("*.xlsb"))) == 10:
            return cand
    return candidates[0]


DATA_DIR = _resolve_data_dir()
OUTPUT_DIR = ROOT_DIR / "solution_outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

SEED = 42
# \u9644\u4ef6 5 (Heat 5: 6C02057) \u5b9e\u8bc1\u6b65\u957f\u4e3a 1.0 s\uff0c\u5176\u4f59\u9644\u4ef6\uff08\u542b\u65b0\u7248\u9644\u4ef6 1: 6B03205\uff09\u5747\u4e3a 0.5 s
HEAT_DT: Dict[int, float] = {k: (1.0 if k == 5 else 0.5) for k in range(1, 11)}


@dataclass(frozen=True)
class PhysicsConstants:
    """\u8f6c\u7089\u70bc\u94a2\u8fc7\u7a0b\u7269\u7406\u4e0e\u5149\u5b66\u5e38\u6570\u3002"""
    adc_saturation: float = 65530.0     # USB4000 16-bit CCD \u53f3\u5220\u5931\u9971\u548c\u9608\u503c (counts)
    adc_rec_upper: float = 105000.0     # \u9971\u548c\u91cd\u6784\u7269\u7406\u8f90\u5c04\u8f6f\u4e0a\u754c (counts)
    hc_over_kb: float = 0.01438777      # \u7b2c\u4e8c\u8f90\u5c04\u5e38\u6570 c2 = hc/k_B (m\xb7K)
    dark_wl_max: float = 385.0          # \u7d2b\u5916\u6697\u7535\u6d41\u57fa\u7ebf\u6807\u5b9a\u6ce2\u6bb5\u4e0a\u9650 (nm)
    common_wl_min: float = 380.0        # 6B/6C \u516c\u5171\u5bf9\u9f50\u6ce2\u957f\u4e0b\u9650 (nm)
    common_wl_max: float = 1010.0       # 6B/6C \u516c\u5171\u5bf9\u9f50\u6ce2\u957f\u4e0a\u9650 (nm)
    common_wl_points: int = 127         # \u516c\u5171\u6ce2\u957f\u7f51\u683c\u70b9\u6570 (5.0 nm \u6b65\u957f)
    typical_o2_total: float = 3400.0    # \u5386\u53f2\u53c2\u8003\u603b\u8017\u6c27\u6807\u5ea6 (m\xb3\uff0c\u6298\u5185\u52a8\u6001\u7531\u8bad\u7ec3\u96c6\u5747\u503c\u66ff\u4ee3)
    o2_gap_jump_m3: float = 14.5        # \u5355\u884c\u8017\u6c27\u5f02\u5e38\u8df3\u53d8\u9608\u503c (m\xb3\uff0c\u9ad8\u4e8e ~10 m\xb3 \u53cc\u8109\u51b2\u62cd\u9891\uff0c\u7cbe\u51c6\u6355\u83b7\u65ad\u884c\u8865\u8d26\u8df3\u53d8)
    max_phys_o2_rate: float = 8.5       # \u6c27\u67aa\u7269\u7406\u6700\u5927\u4f9b\u6c27\u6d41\u91cf\u4e0a\u754c (m\xb3/s)


@dataclass(frozen=True)
class Q1Config:
    """\u7b2c1\u9898\uff1a\u5bf9\u6570\u57df\u53f3\u5220\u5931\u4fdd\u754c\u4f4e\u79e9 SVD \u5149\u8c31\u6062\u590d\u3001\u5bf9\u6570\u4e8c\u5143\u6837\u6761\u66f2\u9762\u4e0e\u5408\u6210\u6ce8\u5165\u9a8c\u8bc1\u53c2\u6570\u3002"""
    spike_temporal_win: int = 5
    spike_spectral_win: int = 5
    spike_z_t: float = 10.0
    spike_z_l: float = 6.0
    frame_med_win: int = 11
    frame_z_thresh: float = 4.5
    frame_log_floor: float = 0.65
    svd_rank: int = 12
    svd_iters: int = 6
    svd_shrink: float = 0.3
    q3_modal_rank: int = 8              # \u6298\u5185\u4ece Q1 \u8fde\u7eed\u5149\u8c31\u63d0\u53d6\u5e76\u4f20\u9012\u81f3 Q3 \u5728\u7ebf\u6295\u5f71\u7684\u4f4e\u79e9\u6a21\u6001\u7ef4\u6570
    sg_wl_win: int = 7                  # ~2.2 nm \u7a97\u53e3\uff0c\u4fdd\u771f 3.4 nm \u95f4\u8ddd\u7684 K I \u53cc\u5cf0
    sg_wl_poly: int = 3
    sg_t_win: int = 7
    sg_t_poly: int = 2
    masked_cv_frac: float = 0.05        # 5% \u968f\u673a\u50cf\u7d20\u7559\u51fa\u5bf9\u6bd4\u8bc4\u4f30


@dataclass(frozen=True)
class Q2Config:
    """\u7b2c2\u9898\uff1a\u53bb\u8865\u8d26\u8df3\u53d8\u6d41\u91cf\u91cd\u6784 + \u591a\u521d\u503c\u5168\u5c40 PFR \u53d8\u65f6\u6ede\u8fa8\u8bc6 + \u5168\u53c2\u6570\u8054\u5408\u91cd\u62df\u5408 Bootstrap \u53c2\u6570\u3002"""
    o2_sg_win_s: float = 7.5
    flow_slope_win_s: float = 24.0      # \u7ea6 2 \u4e2a 11 s \u8ba1\u6570\u5668\u62cd\u9891\u5468\u671f\uff0c\u6291\u5236\u62cd\u9891\u4f2a\u5f71
    gas_sg_win_s: float = 7.5
    stuck_plateau_s: float = 10.0       # \u6d3b\u8dc3\u671f\u5185\u8fde\u7eed >10 s \u6052\u5b9a\u89c6\u4e3a\u4eea\u8868\u5361\u6ede
    gas_cliff_thresh: float = 8.0       # \u7ed3\u6784\u6027\u505c\u5439/\u65ad\u5e27\u53f0\u9636\u9608\u503c
    t_mix_fixed_s: float = 5.0          # \u70df\u7f69\u6df7\u5408\u5ba4\u7269\u7406\u65f6\u95f4\u5e38\u6570\u57fa\u51c6\u503c (s)
    t_mix_sens_grid: Tuple[float, ...] = (2.0, 5.0, 10.0)  # T_mix \u7269\u7406\u654f\u611f\u6027\u68c0\u9a8c\u7f51\u683c (s)
    bootstrap_iters: int = 250          # \u6309\u7089\u6709\u653e\u56de\u3010\u5168\u53c2\u6570\u8054\u5408\u91cd\u62df\u5408\u3011Bootstrap \u6b21\u6570
    flow_query_grid: Tuple[float, ...] = (2.35, 3.0, 3.5, 4.0, 4.35, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0)


@dataclass(frozen=True)
class Q3Config:
    """\u7b2c3\u9898\uff1aQ1 \u6298\u5185\u5149\u8c31\u57fa\u5728\u7ebf\u6295\u5f71 + \u7ebf\u6027\u8d8b\u52bf-\u6811\u6b8b\u5dee\u6df7\u5408\u6a21\u578b + \u56e0\u679c\u8f6f\u95e8\u63a7\u591a\u7089\u96c6\u6210 + \u4e25\u683c\u5185\u5c42\u5d4c\u5957 Smith/\u5171\u5f62\u533a\u95f4\u3002"""
    heat_baseline_s: float = 60.0       # \u524d 60 s \u4e25\u683c\u5355\u4fa7\u56e0\u679c\u7089\u6b21\u57fa\u7ebf\u7a97\u53e3\uff08\u96f6\u672a\u6765\u6cc4\u6f0f\uff09
    ema_scales_s: Tuple[float, ...] = (5.0, 25.0, 80.0, 200.0)
    flicker_scale_s: float = 15.0
    ridge_alpha: float = 600.0          # \u6709\u754c\u7269\u7406\u57fa\u7ebf\u6027\u5916\u63a8\u5934\u6b63\u5219\u5316\u5f3a\u5ea6
    linear_blend_co2: float = 0.18      # CO2 \u7ebf\u6027\u8d8b\u52bf-\u6811\u6b8b\u5dee\u6df7\u5408\u6743\u91cd
    linear_blend_co: float = 0.32       # CO \u7ebf\u6027\u8d8b\u52bf-\u6811\u6b8b\u5dee\u6df7\u5408\u6743\u91cd\uff08\u5f3a\u5316\u9ad8\u78b3\u5cf0\u5916\u63a8\u80fd\u529b\uff09
    expert_gate_weight: float = 0.22    # \u57fa\u4e8e\u524d 60 s \u56e0\u679c\u7279\u5f81\u76f8\u4f3c\u5ea6\u7684\u591a\u7089\u4e13\u5bb6\u8f6f\u95e8\u63a7\u878d\u5408\u6743\u91cd
    xgb_depths: Tuple[int, ...] = (3, 4, 5)
    xgb_seeds: Tuple[int, ...] = (42, 102, 202)
    xgb_n_estimators: int = 180
    xgb_lr: float = 0.045
    xgb_reg_lambda: float = 12.0
    smith_tau_s: float = 5.0
    smith_gain_candidates: Tuple[float, ...] = (0.65, 0.75, 0.85, 0.90, 0.95)
    conformal_alpha: float = 0.10       # 90% \u5c40\u90e8\u5c3a\u5ea6\u81ea\u9002\u5e94\u5171\u5f62\u9884\u6d4b\u533a\u95f4
    probe_intervals_s: Tuple[Tuple[str, float | None], ...] = (
        ("ZeroProbe_100pctSave", None),
        ("Pulse_120s_99.6pctSave", 120.0),
        ("Pulse_60s_99.2pctSave", 60.0),
        ("Pulse_30s_98.3pctSave", 30.0),
        ("Pulse_15s_96.7pctSave", 15.0),
        ("Continuous_DualModal", 0.5),
    )


PHYSICS = PhysicsConstants()
Q1_CFG = Q1Config()
Q2_CFG = Q2Config()
Q3_CFG = Q3Config()


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, CACHE_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    """\u914d\u7f6e\u5168\u82f1\u6587\u79d1\u7814\u671f\u520a\u7ea7 Matplotlib \u7edf\u4e00\u6837\u5f0f\u3002"""
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "axes.labelsize": 10.0,
        "axes.titlesize": 11.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.framealpha": 0.88,
    })


def save_figure(fig, stem: str) -> None:
    """\u540c\u6b65\u5bfc\u51fa 300 DPI PNG \u4e0e\u77e2\u91cf PDF\uff0c\u6ee1\u8db3 LaTeX \u6392\u7248\u4e0e\u5feb\u901f\u9884\u89c8\u53cc\u91cd\u9700\u6c42\u3002"""
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
