# 2026 APMCM Problem B — 双模态融合求解代码与冻结结果（v4 最终冻结版）

> **冻结声明（Final Frozen Release）**：本目录下的全部代码、8 组图表（PNG 300 DPI + 矢量 PDF）、20 份结果表（CSV）及下述全部指标均来自**同一次**无告警端到端完整运行（`python solution_v4_verified/solution_code/main.py --stage all`，2026-11-27 冻结）。论文写作、图例核对与表格排版请统一以本冻结版本为准。

## 目录结构

```text
solution_v4_verified/
├── README.md                                   # 本说明文档（与冻结 CSV / 图例 100% 同步）
├── solution_code/                              # 全模块化求解代码（7 个核心模块）
│   ├── config.py                               # 全局物理常数、超参数与自动数据寻址
│   ├── io_loader.py                            # Calamine 高速加载 + Sequence 连续性校验 + SHA-256 缓存键
│   ├── q1_spectral_recon.py                    # Q1: 对数域右删失低秩 SVD + 对数二元样条 3D 曲面 + 折内模态基接口
│   ├── q2_gas_delay.py                         # Q2: 去补账跳变流量 + 归一化多初值 PFR 变时滞 + 250 次全参数重拟合 Bootstrap
│   ├── q3_fusion_predict.py                    # Q3: Q1 折内光谱基在线投影 + 因果核加权线性树混合 + 内层嵌套 Smith/自适应共形区间
│   ├── scientific_verify.py                    # 6 项科学一致性、盲初值合成回收与零泄漏自动校验
│   └── main.py                                 # 一键式全流程入口 (--stage q1|q2|q3|all)
└── solution_outputs/
    ├── figures/                                # 8 组全英文高分辨率 PNG + PDF 矢量双导出图表
    └── tables/                                 # 20 份定量评估表与 6 项自动校验报告
```

## 运行复现

代码自动识别仓库根目录下的 `B_results/data_raw/xlsb/`（或本地 `附件_Data/Data/`）中的 10 份原始 `.xlsb` 文件（附件 1 为更新版 `6B03205`，附件 5 按实证步长 $\Delta t = 1.0\text{ s}$ 处理并附 $0.5\text{ s}$ 敏感性对照）：

```bash
pip install numpy pandas scipy scikit-learn xgboost matplotlib python-calamine
python solution_v4_verified/solution_code/main.py --stage all
```

---

## 冻结实证指标汇总（与 `solution_outputs/tables/` 及图例完全一致）

### 第 1 题：对数域右删失约束低秩光谱重构与连续 3D 曲面
- **5% 随机像素留出重构误差**（`q1_method_comparison.csv`，对数域 RMSE / 相对误差）：
  - H03：本方法 `0.0406`（相对误差 `2.83%`），优于 2D Whittaker 平滑样条（`0.2267`）、时间线性插值（`0.2624`）与 Savitzky-Golay（`0.2701`）。
  - H08：本方法 `0.0354`（相对误差 `2.38%`），优于 2D Whittaker（`0.3231`）、时间线性插值（`0.3269`）与 Savitzky-Golay（`0.3748`）。
- **物理约束与谱线保真**（`q1_spectral_reconstruction_metrics.csv`）：
  - 全部滤波与二元三次样条 `RectBivariateSpline` 在对数辐射域 $\ln(1+I)$ 内完成，3D 连续曲面无负下冲像素（`3d_neg = 0`）。
  - 饱和像素（16-bit CCD 截断）作为右删失观测处理，重构值严格满足 $I_{\text{rec}} \ge 65530.0\text{ counts}$ 且与未饱和侧翼连续过渡；766.5 / 769.9 nm 双线（可能对应 K I）峰谷比平均保真率为 **99.66%**。
- **1% 合成毛刺注入检测**（`q1_injection_test.csv`）：
  - 去共模闪烁时空联合检测器在 H03 / H08 上取得 **F1 = 0.937 / 0.936**（Precision `0.989 / 0.978`，Recall `0.890 / 0.897`）；作为对照，一维时间 Hampel 基线为 **F1 = 0.470 / 0.252**。

### 第 2 题：累积耗氧坐标下的平推流变时滞模型与一致性诊断
- **模型形式与参数标定**（`q2_delay_vs_flowrate_table.csv`、`q2_tmix_identifiability_sensitivity.csv`）：
  - 函数形式由“平推流输运 + 烟罩一阶混合”机理给出，固定物理混合常数 $T_{\text{mix}} = 5.0\text{ s}$ 后，在 $\mathcal{O}(1)$ 归一化参数空间采用 3 组远距初值（`50 / 115 / 195 m³`）执行有界 `L-BFGS-B` 优化，参数由开吹响应标定并由主吹期阶跃事件独立印证：
    $$\Delta V_{\text{pipe}} = 127.14\text{ m}^3,\quad \tau_{\text{pfr}}(Q) = \frac{127.14}{Q},\quad \tau_{\text{eff}}(Q) = 5.0 + \frac{127.14}{Q}\quad (\text{Global } R^2 = \mathbf{0.9757})$$
  - 当典型主吹流量 $Q = 5.0\text{ m}^3/\text{s}$ 时，$\tau_{\text{pfr}} = 25.43\text{ s}$、总有效时延 $\tau_{\text{eff}} = \mathbf{30.43\text{ s}}$（95% CI `[29.67, 31.52] s`）；观测阶跃响应事件（`q2_step_response_delay_events.csv`）独立拟合 $R^2 = 0.742$。
- **不确定性与敏感性分析**：
  - **250 次全参数联合重拟合 Bootstrap**：按炉有放回重采样并联合重拟合全部 5 个自由参数，得 $\Delta V_{\text{pipe}}$ 的 95% 置信区间 **`[123.4, 132.6] m³`**；10 折留一炉独立标定 $\Delta V_{\text{pipe}}^{(-i)}$ 均值 `127.14 m³`、标准差 `0.92 m³`（范围 `[125.97, 128.87] m³`）。
  - **$T_{\text{mix}}$ 敏感性**：$T_{\text{mix}}$ 取 `2.0 / 5.0 / 10.0 s` 时拟合 $R^2$ 几乎不变（`0.9754 / 0.9757 / 0.9760`），对应 $\Delta V_{\text{pipe}}$ 为 `131.94 / 127.14 / 121.96 m³`，说明 $T_{\text{mix}}$ 单独不可辨识，需固定为物理合理值并作敏感性检验。
- **烟气浓度量纲一致性诊断**（`q2_mass_balance_consistency.csv`）：
  - 10 炉表观积分比 $\alpha_{\text{app}} = \int (y_{\text{CO}_2}+0.5 y_{\text{CO}})\,dt / V_{\text{O}_2} = \mathbf{5.92 \pm 0.39}\text{ vol\%}\cdot\text{s/m}^3$（$\text{CV} = 6.5\%$），推断烟气列为体积百分浓度（`vol%`），对应有效稀释烟道流量 $\hat{Q}_{\text{flue}} \approx 16.9\text{ m}^3/\text{s}$。

### 第 3 题：Q1 模态衔接、因果核加权混合与 Smith 双模态软测量（严格外层 LOHO + 内层 8 对 1 嵌套）

以下数据直接来自冻结版 `q3_loho_ablation_summary.csv`（10 炉外层留一炉交叉验证均值）：

| 模型 / 采样模式 | 探头节能率 | $\text{CO}_2$ Mean $R^2$ | $\text{CO}_2$ RMSE | $\text{CO}$ Mean $R^2$ | $\text{CO}$ RMSE |
|---|---:|---:|---:|---:|---:|
| `M0_ProcessOnly`（纯因果工艺基线） | 100.00% | 0.6686 | 2.464 | 0.8600 | 6.460 |
| `M1_SpectraAndQ1Modes`（仅光谱与 Q1 模态） | 100.00% | 0.3763 | 3.203 | 0.4426 | 11.338 |
| `M2_SpecProc_Unaligned`（光谱+工艺，未时延对齐） | 100.00% | 0.3271 | 3.406 | 0.8009 | 7.645 |
| `M3_SpecProc_Aligned`（光谱+工艺+时延对齐） | 100.00% | 0.6627 | 2.443 | **0.8813** | **6.007** |
| `M4_Whole_LinearTree`（全炉池化线性树混合） | 100.00% | 0.6706 | 2.395 | 0.8713 | 6.233 |
| `Grouped_Converter_6B6C`（按转炉炉型硬分组） | 100.00% | 0.5930 | 2.691 | 0.7567 | 7.734 |
| `Grouped_CausalWardCluster`（前 60 s 因果聚类硬分组） | 100.00% | 0.5536 | 2.808 | 0.7171 | 8.622 |
| `Combined_SoftGated_Hybrid`（零探头前 60 s 因果核加权混合） | 100.00% | 0.6577 | 2.445 | 0.8727 | 6.182 |
| `Hybrid_Smith_Pulse_120s_99.6pctSave`（每 120 s 采样一次） | 99.58% | 0.6947 | 2.307 | 0.9084 | 5.279 |
| **`Hybrid_Smith_Pulse_60s_99.2pctSave`**（每 60 s 采样一次） | **99.17%** | **0.7262** | **2.186** | **0.9203** | **4.934** |
| **`Hybrid_Smith_Pulse_30s_98.3pctSave`**（每 30 s 采样一次） | **98.33%** | **0.7686** | **1.978** | **0.9344** | **4.426** |
| `Hybrid_Smith_Pulse_15s_96.7pctSave`（每 15 s 采样一次） | 96.67% | 0.7991 | 1.834 | 0.9431 | 4.117 |
| **`Hybrid_Smith_Continuous_DualModal`**（全时双模态校正） | 0.00% | **0.8187** | **1.740** | **0.9494** | **3.845** |
| `Naive_Lagged_Sensor_Pulse_60s_99.2pctSave`（60 s 未补偿硬件直读） | 99.17% | -0.1678 | 4.566 | 0.7607 | 8.564 |
| `Naive_Lagged_Sensor_Continuous_DualModal`（0.5 s 未补偿硬件直读） | 0.00% | 0.5383 | 2.860 | 0.9160 | 4.868 |

- **过渡区间表现**（`q3_transition_regime_rmse.csv`）：在阶段断点 ±30 s 内，`Combined_SoftGated_Hybrid` 将 $\text{CO}$ RMSE 由 `8.023`（`M0`）降至 `7.550`，平稳段两项指标均有改善；但在流量剧烈跳变前 10% 区间（`Top10pct_dQ_Jump`）纯工艺基线 `M0` 略低。
- **局部尺度自适应 90% 共形预测区间**（`q3_conformal_90pct_coverage.csv`，严格内层 OOF 标定）：
  - `Hybrid_Smith_Continuous`：10 炉平均覆盖率 $\text{CO}_2 = \mathbf{89.75\%}$、$\text{CO} = \mathbf{89.80\%}$（接近 90% 标称水平）。
  - `Hybrid_ZeroProbe`：10 炉平均覆盖率 $\text{CO}_2 = \mathbf{89.69\%}$、$\text{CO} = \mathbf{86.98\%}$。

---

## 6 项自动化校验结果（`verification_audit_report.csv` — 全部 PASS）

1. `1_Finiteness_NonNegativity_SatBound_and_Zero3DUndershoot`: **PASS** (`finite=True, min_sat=65530.0, 3d_neg=0, min_dO2=0.00e+00`)
2. `2_Concentration_Integral_Closure`: **PASS** (`max_endpoint_closure_diff=3.64e-11 volpct*s`)
3. `3_Arrival_Clock_Strict_Monotonicity_No_TimeFolding`: **PASS** (`folded_steps=0, min_da_step=3.5808e-07 s`)
4. `4_Noisy_BlindStart_Synthetic_Delay_Recovery (True=135.0m3)`: **PASS** (`recovered_dV=135.29 m3 from starts [45, 210], abs_err=0.29 m3`)
5. `5_Tamper_Future_Zero_Leakage_Causality_Test (Incl_Q1_Basis)`: **PASS** (`max_feat_diff=0.000000e+00, max_ksrc_diff=0, max_pred_diff=0.000000e+00`)
6. `6_Cross_Heat_Nested_Fold_Zero_Leakage_Test`: **PASS** (`max_train_artifact_diff_under_corrupted_test_heat=0.000000e+00`)
