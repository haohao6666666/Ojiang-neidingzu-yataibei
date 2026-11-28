# 论文插图（15 张，欧拉奖风格）

所有图都放在 `out/`：PDF 用于 LaTeX，PNG 用于预览和 Word。图宽已经按 APMCM 版心 5.76 in 出好，`\includegraphics[width=<宽度>\textwidth]` 里的宽度照下表填写，图中字号就是 7–9 pt。

```bash
python paper_figures/make_all.py          # 重出全部图（约 1 分钟）
```

## 统一规范（仿 2025 D 题欧拉奖 2519935）

- 底色为浅灰紫 `#EAEAF2`，配白色网格；不画边框，不画刻度线。字体用无衬线 Arial，坐标轴名用深蓝 `#1F2A78`。
- 颜色语义全文固定：**CO 鲑红**、**CO₂ 靛蓝**、O₂ 和模型曲线深海军蓝、6B 薰衣草紫、6C 靛蓝；四个吹炼阶段的底纹在 Fig 10 和 Fig 11 中保持一致。
- 图内不写标题，不写解释性句子，只保留数据、坐标轴、简短图例和 1–2 个关键数字。解释写在图注里。
- 所有数字都从冻结结果文件读取，没有手抄。

## 逐图清单

| 图 | 文件 | 宽度 | 数据来源 | 相比旧版的改动 |
|---|---|---|---|---|
| 1 | `fig01_dual_modal_schematic` | 0.7 | 示意图 | **草稿，供胡乘风作底稿**；新画：转炉、氧枪、火焰、光谱仪（real time）、烟道、ΔV_pipe、弯头处的分析仪（≈30 s later） |
| 2 | `fig02_framework` | 1.0 | 示意图 | **草稿，供胡乘风作底稿**；新画：Data → Models → Results 三层色带；Task 1 的光谱基和 Task 2 的时延都输入 Task 3 |
| 3 | `fig03_preprocessing_workflow` | 1.0 | 示意图 | **草稿，供胡乘风作底稿**；新画：5 个步骤，每步只写一个关键结论 |
| 4 | `fig04_o2_counter_beat` | 1.0 | A 版 Heat 3 原始计数 | 统一配色；(b) 的窗口说明改成图例 |
| 5 | `fig05_carbon_oxygen_balance` | 0.7 | A 版 `q2_mass_balance.csv` | 去掉标题；均值线和物理上限放进图例 |
| 6 | `fig06_arrival_time_map` | 0.8 | PFD 拟合值（v4） | 新画：上层为映射关系，下层把时延分解为输运加混合（59.1 s / 30.4 s，与 Table 6 一致） |
| 7 | `fig07_rlr_decomposition` | 1.0 | A 版 Heat 3 缓存 | 改用统一色图；在稀疏项上标出 K 767 nm 和 Na 589 nm |
| 8 | `fig08_outlier_reconstruction` | 1.0 | A 版 Heat 9 缓存 | 旧图右栏选到的是 t = 0 的帧，并不异常；改为 9.14 min 的饱和亮帧（64% 像素被标记） |
| 9 | `fig09_spectral_surfaces` | 1.0 | A 版趋势曲面 | 十炉共用同一 z 轴范围和色条；刻度只在下排标注 |
| 10 | `fig10_process_curves_heat03` | 1.0 | A 版 Heat 3 | 加四阶段底纹；**单位由 m³/s 改为 vol.%，与 Assumption 4 一致** |
| 11 | `fig11_four_stage_relation` | 1.0 | A 版十炉 | 加阶段底纹，并标出断点 |
| 12 | `fig12_plug_flow_delay` | 1.0 | v4 重跑导出 | **右栏改为留一炉标定的 ΔV_pipe**（旧右栏与 Fig 5 是同一组数据） |
| 13 | `fig13_windowed_delay_vs_flow` | 0.7 | A 版 44 个窗口 | **曲线已换成 PFD τ = 5 + 127.1/Q**（文稿要求）；标注 Spearman ρ = −0.75 |
| 14 | `fig14_loho_predictions` | 1.0 | v4 重跑导出 | **改画零探头 DSS**（旧图画的是连续 Smith 校正，与正文的数字对不上）；Heat 1 用浅色底标出 |
| 15 | `fig15_transfer_probe_vip` | 1.0 | v4 表格 | 迁移矩阵的对角线置灰；(b) 高亮 120 s 列；(c) 标出 K/Na 线和 730–825 nm 波段 |

## 图注（已同步写入 `B题论文文稿.md`）

- **Fig 8**：注明 Heat 9。Right: an anomalous bright frame at 9.14 min (Heat 9), rebuilt from its retained pixels.
- **Fig 10**：CO and CO₂ in vol.%; cumulative emissions in 10³ vol.%·s.
- **Fig 12**：Right: leave-one-heat-out recalibration of ΔV_pipe (127.1 ± 0.9 m³) inside the full-data 95% bootstrap interval.（替换原来的 "apparent scale factor"）
- **Fig 14**：LOHO zero-probe DSS predictions (dashed) and 90% conformal intervals (bands) against the delay-aligned readings (solid).
- **Fig 15 (b)**：横轴只标采样间隔，图注补一句 "120, 60, 30, 15 s save 99.6%, 99.2%, 98.3%, 96.7% of probe readings."

## 数据与复现

- `extract_data.py`：从 `B_results/data_clean` 缓存、两版冻结表格，以及 v4 代码重跑的导出数据，生成 `data/` 下的紧凑数据。
- v4 重跑结果与冻结表**完全一致**：18 行汇总的最大差值为 0，逐炉 CO R² 与正文相同。

## Fig 1–3 草稿说明（给胡乘风）

这三张示意图是底稿，可以直接放进论文，也可以照着用 TikZ 或 draw.io 重画。改文字或配色时，直接编辑对应脚本（`fig01_schematic.py`、`fig02_framework.py`、`fig03_preprocessing.py`）后运行即可；圆角框和箭头的画法在 `diagram.py` 里，颜色在 `style.py` 里。重画时请保持：

- 配色与数据图一致：靛蓝 `#4F6096`、薰衣草紫 `#B3A6CF`、桃 `#F4C38F`、鲑红 `#E0876A`、深蓝文字 `#1F2A78`；
- 每个框只放短词或一个关键数字，解释写进图注；
- Fig 2 中 Task 1 → Task 3（spectral basis）和 Task 2 → Task 3（delay）两条依赖箭头必须保留，这是"各问之间的逻辑"。
