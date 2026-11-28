# B 题论文英文润色说明（给王壹）

文稿：仓库根目录 [`B题论文文稿.md`](B题论文文稿.md)（英文正文 v2，摘要 466 词，正文约 6,300 词）。
本轮任务：**只润色英文表达，不改模型、数字和结论**。文本冻结时间：**11/29 12:00**，之后交给胡乘风排版。

---

## 1. 拉取与提交

```bash
git pull origin main
```

- 直接在 `B题论文文稿.md` 上改，**每改完一节就提交并推送一次**（例如 `git commit -m "Polish Section 6 wording"`），避免和倪浩宇的修改冲突。
- 推送前先 `git pull --rebase origin main`；如有冲突，以对方已推送的版本为底再合入自己的改动。
- 拿不准的地方不要硬改，在句后加标记 `[[WY: 你的疑问或备选写法]]`，倪浩宇统一处理。

## 2. 不能动的内容

| 内容 | 原因 |
|---|---|
| 所有数字（R²、区间、体积、时延、百分比、炉号、方程编号、图表编号） | 全部来自冻结结果（v4 提交 `416495a` 与 `B_results/tables/`），每节末尾“数字出处”可查 |
| 公式（`$...$`、`$$...$$`）及符号 | 与 Notation 表一一对应 |
| 结论的方向和强度 | shows / indicates / suggests 已按证据强度选定；不要把 suggests 改成 shows，也不要加 "significantly"、"optimal"、"proves" |
| 局限与上界的表述 | 如 "bound the transport delay from above"、Weaknesses 三条，删掉会变成过度声称 |
| 中文方框：使用说明、`[Figure k]` 图位、“数字出处” | 排版时由胡乘风删除，润色时跳过即可 |
| 章节结构与标题编号 | 已与图表位置对齐；标题措辞可以提建议，用 `[[WY: ...]]` 标注 |

**篇幅不要再加长。**正文已在 25 页上限边缘，润色以“同义替换、拆长句、删冗词”为主，净字数最好持平或略减。

## 3. 重点润色位置（按优先级）

1. **Summary**（评委第一眼看）：每句是否一读就懂；四个加粗 Task 段的洞见句是否有力。
2. **每问开头的问题分析段**（5、6、7 节第一段）与 **1.3 Our Work**：逻辑连接词是否清楚。
3. **结果解读段**（5.2、6.4、7.3）：看“现象（数字）→ 原因 → 意义”是否顺。
4. **9 Strengths/Weaknesses、10 Conclusions**：句子要短、要具体。
5. 图注（Caption 行）：名词短语 + 一句结论，保持这个格式。

## 4. 语言规范（O 奖写法）

- 用 **we**，不用 "this paper"、"the authors"、"I"。
- 描述模型和结论用**一般现在时**；描述已做的数据处理可用过去时；同一段不要来回切换。
- 优先**主动语态**；一句一个意思，超过 30 词考虑拆分；不要用缩写（cannot 而非 can't）。
- 不以数学符号开头一句；带单位的数字用阿拉伯数字（5 s），作形容词的小数字用单词（three heats）。
- 不用 AI 腔和空话（“robust”作为方法术语，如 robust low-rank、robust σ，要保留）：delve, leverage, crucial, pivotal, comprehensive, seamless, notably, "It is worth noting that", "plays a vital role", "In conclusion,"。
- 不要每段都以 "As shown in Figure" 开头，让结论当主语：`The delay falls as the flow rises (Figure 13).`
- 不加防御性语言（"to some extent", "it can be seen that", "we believe"）和口语（"a lot of", "pretty", "basically"）。

## 5. 给 GPT 的提示词（可直接复制，一次只贴一节）

```text
You are editing one section of an English mathematical modelling contest paper (APMCM, aiming for a first prize).
Polish the language only. Rules:
1. Do not change any number, formula, symbol, figure/table/equation number, or citation.
2. Do not change the meaning or the strength of any claim (keep shows/indicates/suggests as they are); do not add new claims, data or references.
3. Keep limitations and "upper bound" statements.
4. Use "we", present tense for models and findings, active voice, sentences under 30 words, no contractions.
5. Remove filler and AI-style words (delve, leverage, crucial, pivotal, comprehensive, notably, "it is worth noting").
6. Do not make the text longer; aim for the same or fewer words.
7. Leave every line that starts with ">" and any Chinese text unchanged.
Return the full revised section in Markdown, then a short list of the changes you made.

[paste one section here]
```

GPT 改完后**逐句对照原文**再贴回去，特别注意它是否悄悄改了数字或把限定词删掉了。

## 6. 提交前自检（两条命令）

```bash
python3 tools/check_numbers.py
```
对比上一次提交，列出出现次数变化的数字。输出 `OK` 才算数字没动；若列出某个数字，回到原文核对（改写时合并掉一个重复数字可以，改了数值绝对不行）。

```bash
python3 tools/lint_english.py B题论文文稿.md
```
英文风格检查：长句、缩写、过度措辞、AI 腔短语、连续同一开头等。每条提示都要结合上下文判断，不是必须替换。表格行、中文说明和列表项被提示属于正常情况。

## 7. 交付

- 11/28 晚上前完成第一遍并推送，倪浩宇 11/29 上午复核。
- 推送后在群里说一声改到了哪一节，以及所有 `[[WY: ...]]` 标记的数量。
