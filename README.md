# 2026 APMCM Problem B: Dual-modal temporal fusion for converter steelmaking

本仓库只包含 B 题相关内容。

| 路径 | 内容 |
|---|---|
| `problem/` | B 题题目 PDF |
| `B题最终合并方案.md` | **最终合并方案**：两版（B_results 与 solution_v4_verified）每问的取舍、引用数字与来源、措辞要求、正文图表清单、待修正问题与分工 |
| `B题建模与算法设计思路.md` | 建模与算法设计（v2） |
| `B_results/结果说明.md` | 三问的方法、关键数字、需人工判断之处 |
| `B_results/code/` | 全部代码，`run_all.sh` 一键复现（约 25 分钟） |
| `B_results/figures/` | 全部图（PNG 300 dpi + PDF，英文标注） |
| `B_results/tables/` | 全部结果表（csv） |
| `B_results/data_raw/xlsb/` | 10 个炉次的原始数据。附件 1 为补全烟气数据后的新版；官方 Data.7z 中的旧版附件 1（CO/CO2 全为 0）未上传 |
| `solution_v4_verified/` | **最终冻结版（v4，2026-11-27 单次端到端运行冻结）**：完整 7 模块源码、8 组 PNG+PDF 图表、20 份评估表及 6 项零泄漏校验报告（图例、表格与文档口径 100% 统一；一键 `python solution_v4_verified/solution_code/main.py --stage all`） |

运行环境：Python 3（numpy、scipy、pandas、scikit-learn、lightgbm、matplotlib）与 LibreOffice（`soffice`，用于把 .xlsb 转为 csv）。
中间缓存（csv、npz、data_clean）未上传，运行 `B_results/code/run_all.sh` 会自动重新生成。
