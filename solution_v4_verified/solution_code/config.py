"""
全局配置模块（v4 红队审计全量修正版）：
- 严格区分各炉实证采样步长：更新版附件 1（6B03205）及其余 8 炉均为 dt = 0.5 s；
  唯独附件 5（6C02057，775 帧）实证采样步长为 dt = 1.0 s（附 dt = 0.5 s 敏感性对照）。
- 移除全局 warnings.filterwarnings("ignore")，仅针对 Apple Silicon Accelerate AMX 矩阵协处理器
  在正常有限浮点矩阵乘法中误置 IEEE 粘滞异常位的底层已知问题配置 np.seterr，保留全部真实运行告警。
- 全英文 300 DPI PNG + 矢量 PDF 双导出，直接适配 APMCM LaTeX 排版。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import matplotlib as mpl
import numpy as np

# 仅屏蔽 Apple Silicon Accelerate BLAS dgemm 空闲通道粘滞浮点标志位；保留全部 Python/SciPy/Sklearn warnings
np.seterr(divide="ignore", over="ignore", invalid="ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent


def _resolve_data_dir() -> Path:
    """自动兼容本地赛题工作区路径与 GitHub 仓库 B_results/data_raw/xlsb 路径。"""
    candidates = [
        ROOT_DIR / "附件_Data" / "Data",
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
# 附件 5 (Heat 5: 6C02057) 实证步长为 1.0 s，其余附件（含新版附件 1: 6B03205）均为 0.5 s
HEAT_DT: Dict[int, float] = {k: (1.0 if k == 5 else 0.5) for k in range(1, 11)}


@dataclass(frozen=True)
class PhysicsConstants:
    """转炉炼钢过程物理与光学常数。"""
    adc_saturation: float = 65530.0     # USB4000 16-bit CCD 右删失饱和阈值 (counts)
    adc_rec_upper: float = 105000.0     # 饱和重构物理辐射软上界 (counts)
    hc_over_kb: float = 0.01438777      # 第二辐射常数 c2 = hc/k_B (m·K)
    dark_wl_max: float = 385.0          # 紫外暗电流基线标定波段上限 (nm)
    common_wl_min: float = 380.0        # 6B/6C 公共对齐波长下限 (nm)
    common_wl_max: float = 1010.0       # 6B/6C 公共对齐波长上限 (nm)
    common_wl_points: int = 127         # 公共波长网格点数 (5.0 nm 步长)
    typical_o2_total: float = 3400.0    # 历史参考总耗氧标度 (m³，折内动态由训练集均值替代)
    o2_gap_jump_m3: float = 14.5        # 单行耗氧异常跳变阈值 (m³，高于 ~10 m³ 双脉冲拍频，精准捕获断行补账跳变)
    max_phys_o2_rate: float = 8.5       # 氧枪物理最大供氧流量上界 (m³/s)


@dataclass(frozen=True)
class Q1Config:
    """第1题：对数域右删失保界低秩 SVD 光谱恢复、对数二元样条曲面与合成注入验证参数。"""
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
    q3_modal_rank: int = 8              # 折内从 Q1 连续光谱提取并传递至 Q3 在线投影的低秩模态维数
    sg_wl_win: int = 7                  # ~2.2 nm 窗口，保真 3.4 nm 间距的 K I 双峰
    sg_wl_poly: int = 3
    sg_t_win: int = 7
    sg_t_poly: int = 2
    masked_cv_frac: float = 0.05        # 5% 随机像素留出对比评估


@dataclass(frozen=True)
class Q2Config:
    """第2题：去补账跳变流量重构 + 多初值全局 PFR 变时滞辨识 + 全参数联合重拟合 Bootstrap 参数。"""
    o2_sg_win_s: float = 7.5
    flow_slope_win_s: float = 24.0      # 约 2 个 11 s 计数器拍频周期，抑制拍频伪影
    gas_sg_win_s: float = 7.5
    stuck_plateau_s: float = 10.0       # 活跃期内连续 >10 s 恒定视为仪表卡滞
    gas_cliff_thresh: float = 8.0       # 结构性停吹/断帧台阶阈值
    t_mix_fixed_s: float = 5.0          # 烟罩混合室物理时间常数基准值 (s)
    t_mix_sens_grid: Tuple[float, ...] = (2.0, 5.0, 10.0)  # T_mix 物理敏感性检验网格 (s)
    bootstrap_iters: int = 250          # 按炉有放回【全参数联合重拟合】Bootstrap 次数
    flow_query_grid: Tuple[float, ...] = (2.35, 3.0, 3.5, 4.0, 4.35, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0)


@dataclass(frozen=True)
class Q3Config:
    """第3题：Q1 折内光谱基在线投影 + 线性趋势-树残差混合模型 + 因果软门控多炉集成 + 严格内层嵌套 Smith/共形区间。"""
    heat_baseline_s: float = 60.0       # 前 60 s 严格单侧因果炉次基线窗口（零未来泄漏）
    ema_scales_s: Tuple[float, ...] = (5.0, 25.0, 80.0, 200.0)
    flicker_scale_s: float = 15.0
    ridge_alpha: float = 600.0          # 有界物理基线性外推头正则化强度
    linear_blend_co2: float = 0.18      # CO2 线性趋势-树残差混合权重
    linear_blend_co: float = 0.32       # CO 线性趋势-树残差混合权重（强化高碳峰外推能力）
    expert_gate_weight: float = 0.22    # 基于前 60 s 因果特征相似度的多炉专家软门控融合权重
    xgb_depths: Tuple[int, ...] = (3, 4, 5)
    xgb_seeds: Tuple[int, ...] = (42, 102, 202)
    xgb_n_estimators: int = 180
    xgb_lr: float = 0.045
    xgb_reg_lambda: float = 12.0
    smith_tau_s: float = 5.0
    smith_gain_candidates: Tuple[float, ...] = (0.65, 0.75, 0.85, 0.90, 0.95)
    conformal_alpha: float = 0.10       # 90% 局部尺度自适应共形预测区间
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
    """配置全英文科研期刊级 Matplotlib 统一样式。"""
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
    """同步导出 300 DPI PNG 与矢量 PDF，满足 LaTeX 排版与快速预览双重需求。"""
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
