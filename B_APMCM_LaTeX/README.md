# 2026 APMCM B题 LaTeX 工程

英文底稿取自团队仓库的 `51d2085cca89cb0090969f6ea46fc53b57d7cc3c` 提交。项目以当届 `apmcmthesis` 类为基础，采用 MCMICM-skill 的欧拉奖配色，并参照 2025 年 MCM/ICM D 题 2519935 样稿排版。

- `main.tex` 是入口，`team-number.tex` 保存正式队号。
- `sections/` 下分别是摘要、正文、参考文献和附录，可直接编辑。
- 第1页和第2页使用2026 APMCM模板的摘要页与目录页结构。第3张起为正文，按官方模板从第1页编号。目录显示到三级，带点线、页码和附录条目。
- 校样默认在参考文献结束。正式提交如需恢复逐炉图表和软件与源码附录，在 `main.tex` 将 `\paperappendixfalse` 改为 `\paperappendixtrue`。
- 正文插图基于团队仓库的 `c16d95b62d6ac9ba04d478d3c4bddbbc1b6de39f` 提交。图3图注写明五步预处理。图14标记样本终点，图15(b)用下箭头注明超出显示范围的R²。
- `figure-source/` 保留图1、6、7、14、15的重绘脚本和所用的冻结输入。在工程目录用 Python 运行对应脚本会更新 `figures/` 中的PDF。需要 NumPy、pandas 和 Matplotlib。
- 参考文献按首次引用顺序排列。重新运行 `prepare_paper.py` 会从原稿覆盖生成的章节，请先备份人工修改。
- `source-listings/` 是附录使用的源码文本。非ASCII字符显示为Unicode转义，便于排版。

在本目录运行两次 `xelatex main.tex`，或运行 `latexmk -xelatex main.tex`。在 Overleaf 中将编译器设为 XeLaTeX。正文、图、表和引用均已排入本项目，编译不需要外部仓库或竞赛数据。

报名号尚未提供。使用正式队号替换 `team-number.tex` 中的占位文本。

打开 [PDF 校样](APMCM-B-preview.pdf) 查看最新版。校样共25页，图15在PDF第22页，三个子图自上而下排列，分别附有图内说明。
