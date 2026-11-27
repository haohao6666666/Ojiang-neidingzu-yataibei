# 2026 APMCM Problem B — 完整双模态融合求解方案（v4 红队审计全量修正版）

本目录包含 2026 APMCM B 题（转炉炼钢过程双模态时序信息融合建模）的全流程求解代码、全套出版级 300 DPI PNG / 矢量 PDF 图表、20 份定量评估与敏感性 CSV 表格，以及 6 项自动化科学一致性与双零泄漏（未来帧零泄漏 + 跨炉零泄漏）审计报告。

## 目录结构

```text
solution_v4_verified/
├── README.md                                   # 本说明文档
├── solution_code/                              # 全模块化求解代码（7 个核心模块）
│   ├── config.py                               # 全局物理常数、超参数与自动数据寻址
│   ├── io_loader.py                            # Calamine 高速加载 + Sequence 连续性校验 + SHA-256 缓存键
│   ├── q1_spectral_recon.py                    # Q1: 对数域右删失低秩 SVD + 对数二元样条 3D 曲面 + 折内模态基接口
│   ├── q2_gas_delay.py                         # Q2: 去补账跳变流量 + 多初值全局 PFR 变时滞 + 250 次全参数重拟合 Bootstrap
│   ├── q3_fusion_predict.py                    # Q3: Q1 折内光谱基在线投影 + 因果核加权线性树混合 + 内层嵌套 Smith/自适应共形区间
│   ├── scientific_verify.py                    # 6 项单元测试级科学一致性、盲初值合成回收与双零泄漏自动审计
│   └── main.py                                 # 一键式全流程入口 (--stage q1|q2|q3|all)
└── solution_outputs/
    ├── figures/                                # 8 组全英文高分辨率 PNG + PDF 矢量双导出图表
    └── tables/                                 # 20 份完整量化评估表与 6 项自动审计报告
```

## 一键复现

代码会自动识别仓库根目录下的 `B_results/data_raw/xlsb/`（或本地 `附件_Data/Data/`）中的 10 份原始 `.xlsb` 文件（附件 1 为更新版 `6B03205`，附件 5 按实证步长 $\Delta t = 1.0\text{ s}$ 处理并附 $0.5\text{ s}$ 敏感性对照）：

```bash
pip install numpy pandas scipy scikit-learn xgboost matplotlib python-calamine
python solution_v4_verified/solution_code/main.py --stage all
```

## 核心实证指标汇总（新版附件 1 全量验证）

### 第 1 题：对数域右删失约束低秩光谱重构与 3D 曲面
- **1% 随机像素留出重构**：10 炉平均 $R^2 = \mathbf{0.9990}$（最低 $0.9987$），对数域 5% 留出误差显著优于时间线性插值、Savitzky-Golay 与 2D Whittaker 平滑样条（`q1_method_comparison.csv`）。
- **物理保界与零黑洞**：全部平滑与 `RectBivariateSpline` 严格在对数辐射域完成，3D 曲面负下冲像素严格为 `0`；饱和帧严格保持右删失物理下界 $I_{\text{rec}} \ge 65530.0\text{ counts}$ 且未饱和侧翼无缝连续；K I 双线（766.5 / 769.9 nm）峰谷比平均保真率 **99.66%**。
- **合成异常注入检测**：去共模闪烁时空联合检测器在 H03 / H08 上取得 **F1 = 0.991 / 0.967**（一维 Hampel 基线仅 0.342 / 0.126）。

### 第 2 题：变流量平推流（PFR）时滞辨识与全参数重拟合 Bootstrap
- **全局动力学耦合辨识**：固定物理混合常数 $T_{\text{mix}} = 5.0\text{ s}$（附 $T_{\text{mix}} \in \{2, 5, 10\}\text{ s}$ 敏感性表），远离先验的 3 初值全局 `L-BFGS-B` 唯一收敛至：
  $$\Delta V_{\text{pipe}} = 126.91\text{ m}^3,\quad \tau_{\text{pfr}}(Q) = \frac{126.91}{Q},\quad \tau_{\text{eff}}(Q) = 5.0 + \frac{126.91}{Q}\quad (\text{Global } R^2 = \mathbf{0.9755})$$
- **250 次全参数联合重拟合 Bootstrap**：按炉有放回重采样并联合重拟合全部 5 个自由参数，得 $\Delta V_{\text{pipe}}$ 的 95% 置信区间 **`[118.9, 136.5] m³`**；10 折留一炉标定稳定性 `127.26 ± 1.86 m³`。
- **拉曼浓度标度物理一致性**：辨识跨炉表观积分标度因子 $\alpha_{\text{app}} = \int (y_{\text{CO}_2}+0.5 y_{\text{CO}})\,dt / V_{\text{O}_2} = \mathbf{5.92 \pm 0.39}\text{ vol\%}\cdot\text{s/m}^3$（$\text{CV} = 6.5\%$），对应有效稀释烟道流量 $\hat{Q}_{\text{flue}} \approx 16.9\text{ m}^3/\text{s}$。

### 第 3 题：Q1 模态衔接、因果核加权线性树混合与双模态 Smith 软测量（严格外层 LOHO + 内层 8-vs-1 嵌套）

| 模型 / 采样模式 | 探头节能率 | $\text{CO}_2$ Mean $R^2$ | $\text{CO}_2$ RMSE | $\text{CO}$ Mean $R^2$ | $\text{CO}$ RMSE |
|---|---:|---:|---:|---:|---:|
| `M0_ProcessOnly`（纯因果工艺基线） | 100.0% | 0.6703 | 2.459 | 0.8594 | 6.475 |
| `Grouped_Converter_6B6C`（按炉型硬分组） | 100.0% | 0.5950 | 2.680 | 0.7544 | 7.760 |
| `Grouped_CausalWardCluster`（前60s因果聚类硬分组） | 100.0% | 0.5576 | 2.793 | 0.7143 | 8.656 |
| `Combined_SoftGated_Hybrid`（零探头因果核加权混合） | 100.0% | 0.6653 | 2.426 | 0.8760 | 6.107 |
| **`Hybrid_Smith_Pulse_60s`**（60 s 脉冲双模态软测量） | **99.17%** | **0.7313** | **2.171** | **0.9227** | **4.862** |
| **`Hybrid_Smith_Pulse_30s`**（30 s 脉冲双模态软测量） | **98.33%** | **0.7733** | **1.967** | **0.9373** | **4.334** |
| **`Hybrid_Smith_Continuous_DualModal`**（全时双模态） | 0.00% | **0.8240** | **1.720** | **0.9523** | **3.741** |
| `Naive_Lagged_Sensor_Pulse_60s`（60 s 未补偿硬件直读） | 99.17% | -0.1658 | 4.567 | 0.7601 | 8.571 |
| `Naive_Lagged_Sensor_Continuous`（0.5 s 未补偿硬件直读） | 0.00% | 0.5387 | 2.860 | 0.9155 | 4.877 |

- **严格内层 OOF 90% 局部自适应共形区间**：`Hybrid_Smith_Continuous` 在 10 炉上取得平均覆盖率 **$\text{CO}_2 = 89.76\%$ / $\text{CO} = 90.03\%$**（精准贴合 90% 标称水平）。

### 6 项自动化科学一致性与双零泄漏审计（`verification_audit_report.csv` — 全部 PASS）
1. `1_Finiteness_NonNegativity_SatBound_and_Zero3DUndershoot`: **PASS** (`finite=True, min_sat=65530.0, 3d_neg=0`)
2. `2_Concentration_Integral_Closure`: **PASS** (`max_endpoint_closure_diff=3.64e-11 volpct*s`)
3. `3_Arrival_Clock_Strict_Monotonicity_No_TimeFolding`: **PASS** (`folded_steps=0, min_da_step=3.5808e-07 s`)
4. `4_Noisy_BlindStart_Synthetic_Delay_Recovery (True=135.0m3)`: **PASS** (`recovered_dV=135.07 m3 from starts [45, 210], abs_err=0.07 m3`)
5. `5_Tamper_Future_Zero_Leakage_Causality_Test (Incl_Q1_Basis)`: **PASS** (`max_feat_diff=0.000000e+00, max_ksrc_diff=0, max_pred_diff=0.000000e+00`)
6. `6_Cross_Heat_Nested_Fold_Zero_Leakage_Test`: **PASS** (`max_train_artifact_diff_under_corrupted_test_heat=0.000000e+00`)
