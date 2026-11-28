# B 题论文文稿（英文正文，v2：扩充至约 6,300 词，按 O 奖写作规范全文润色）

> 使用说明（这一部分不进入论文）
> - 本文稿只写文字，不做 LaTeX/Word 排版。每处图表用方框标注：**图号、文件路径、建议宽度、图注**。标注“PLACEHOLDER”的图尚未绘制，排版时先放占位框。
> - 所有数字取自冻结结果：A 版 `B_results/tables/`，B 版 `solution_v4_verified/solution_outputs/tables/`（提交 `416495a`），以及本轮补算的 `B_results/tables/q3_delay_misestimation_sensitivity.csv`。每节末尾的“数字出处”只供核对，不进入论文。
> - 篇幅：摘要 466 词；正文（引言至结论）6,321 词，另有图注约 490 词与 9 张表；15 张图。按 APMCM 模板粗估 25–27 页，**有超出 25 页上限的风险**。排版时按顺序处理：① 正文用 11 pt（规范允许 11 或 12 pt）；② 仍超页则把 Table 2、Figure 3、Figure 8 移入附录（正文保留一句引用）；③ 再超则把 Table 9 压缩为 5 行。
> - 参考文献共 13 篇，全部已逐条核实 DOI 与真实出版信息（含 10 篇经典方法文献与 3 篇转炉火焰光谱/烟气控制/工业软测量领域文献）。

---

# Seeing the Off-Gas Before It Arrives: Delay-Aware Fusion of Flame Spectra and Flue-Gas Signals for Converter Steelmaking

## Summary

In converter steelmaking, the mouth flame reflects bath reactions in real time, whereas the flue-gas analyzer reports them only after the gas crosses the duct, about half a minute later. We develop a **Delay-Aware Dual-Modal Framework** that cleans both streams, quantifies the transport delay and infers the off-gas state from the flame before the analyzer measures it.

Before modelling, we diagnose four data anomalies. We adopt the corrected Attachment 1 and establish a sampling interval of 1 s for Heat 5. Because the cumulative-oxygen counter advances in steps of about 4.73 m³ with a double step every 11 s, we estimate the lance flow from a 24-s regression slope rather than direct differencing. A carbon–oxygen balance shows that, read as flow rates, the CO and CO₂ values would require **5.5–6.5 times** the supplied oxygen; we therefore interpret them as volume fractions.

**Task 1: Spectral cleaning and continuity.** We build a **Rank-Constrained Robust Low-Rank model (RLR)** that separates each log spectrum into three components and sparse outliers. The first component tracks flame brightness and the second tracks spectral colour. In an injection test, RLR detects spikes with an F1 score of **0.89** (Hampel filter: 0.22), and the continuous surface reconstructs held-out pixels with a relative error of **2.0%**. Sparse outliers concentrate on the potassium and sodium lines, which fluctuate independently of the thermal continuum.

**Task 2: Oxygen, gas and delay.** We reconstruct six continuous process curves per heat and identify four blowing stages, with breakpoints at **23%, 43% and 90%** of the total oxygen (piecewise R² = 0.956–0.997). In the cumulative-oxygen coordinate, the gas trajectories of all ten heats collapse onto one curve, which the **Plug-Flow Delay model (PFD)** fits with R² = 0.976. The model gives **τ(Q) = 5 + 127.1/Q s**, so the delay drops from 47.4 s at 3 m³/s to **30.4 s** at 5 m³/s. Windowed cross-correlation independently confirms that the delay decreases with the flow (Spearman ρ = −0.75, p = 3.4×10⁻⁹).

**Task 3: Real-time prediction.** We build a **Dual-Modal Soft Sensor (DSS)** that projects each frame onto a low-rank spectral basis learned across the training heats. It aligns the prediction target with the PFD arrival map and combines heat-level models through similarity weighting. Under leave-one-heat-out validation across all ten heats, the zero-probe DSS predicts CO and CO₂ with mean R² of **0.873** and **0.658**. A Smith-type correction with one probe reading every 120 s (**99.6% fewer readings**) raises R² to **0.908** and **0.695**, about 30 s ahead of the analyzer.

Sensitivity analysis shows that a ±10% error in the duct volume shifts the zero-probe R² by at most 0.003, and that the stage and delay results hold across both converter groups. Flame spectra can thus replace most probe readings while delivering off-gas information earlier.

**Keywords:** robust low-rank decomposition; plug-flow delay; soft sensor; leave-one-heat-out validation; Smith predictor; converter steelmaking

---

## Contents

（排版时自动生成，列到三级标题。）

---

## 1 Introduction

### 1.1 Background

The basic oxygen converter turns hot metal into steel by blowing high-purity oxygen through a lance at supersonic speed. Silicon, manganese, phosphorus and carbon are oxidized in turn, and the carbon leaves the bath as CO and CO₂. Operators must time the end of the blow so that the carbon content and temperature of the steel reach their targets together [11]. They therefore follow the progress of decarburization from two sources. A spectrometer at the converter mouth records the flame radiation every 0.5 s, and a Raman analyzer at the flue-gas bend records the CO and CO₂ contents at the same rate.

The two sources see the same reactions at different times. The flame responds to the bath immediately, while the gas must travel through the hood and the duct before the analyzer can measure it. This transport delay limits endpoint control, because a decision based on the analyzer refers to a bath state that is already half a minute old. The sampling probe also wears quickly in the dusty, high-temperature duct, so frequent sampling is costly. A data-driven soft sensor that reads the off-gas state from the flame [12, 13] would provide the signal earlier and would allow the probe to work less often.

> **[Figure 1]** `paper_figures/out/fig01_dual_modal_schematic.pdf`，宽度 0.7。（**草稿，供胡乘风作底稿**：可直接使用，或据此用 TikZ/draw.io 重画；可编辑源文件 `paper_figures/fig01_schematic.py`）
> Caption: **Dual-modal measurement at the converter.** The spectrometer observes the flame in real time; the analyzer observes the same reactions after the gas has crossed the duct volume $\Delta V_{\mathrm{pipe}}$.

### 1.2 Restatement of the Problem

The problem provides the full-process flame spectra (2,047 wavelengths, 0.5-s step), the cumulative oxygen consumption and the CO and CO₂ signals of ten heats. We address the following tasks.

- **Task 1.** Remove the outliers in the spectral data, reconstruct a continuous spectral surface, and draw the three-dimensional spectra of all ten heats.
- **Task 2.** Remove the outliers in the oxygen and gas series, reconstruct continuous curves of the lance flow, cumulative oxygen, CO, CO₂ and their cumulative emissions, relate the gas signals to the oxygen consumption, and compute the flue-gas delay under different lance flow rates.
- **Task 3.** Use the results of Tasks 1 and 2 to predict the flue-gas information in real time from the flame spectra, and test the prediction on all ten heats.
- **Combination requirement.** Combine the models learned from different heats, as a whole or by groups, so that the combined model performs well on all ten heats.

### 1.3 Our Work

Figure 2 outlines our workflow. We first diagnose the data and resolve four inconsistencies that would otherwise distort every downstream step. Section 4 then introduces an arrival-time map linking each real-time flame observation to the delayed gas reading it generates. The three tasks proceed as an integrated pipeline. First, Task 1 builds a low-rank model of the flame spectra to remove outliers and supply cleaned spectral representations. Second, Task 2 reconstructs the process curves, partitions each blow into chemical stages and quantifies how transport delay varies with lance flow. Finally, Task 3 feeds a spectral basis built on the low-rank structure of Task 1 and the delay map from Task 2 into a causal real-time predictor. It combines the heat-level models and updates predictions whenever a probe reading arrives.

> **[Figure 2]** `paper_figures/out/fig02_framework.pdf`，宽度 1.0。（**草稿，供胡乘风作底稿**：可直接使用，或据此用 TikZ/draw.io 重画；可编辑源文件 `paper_figures/fig02_framework.py`）
> Caption: **Framework of our work.** The low-rank spectral structure of Task 1 and the delay model of Task 2 feed the soft sensor of Task 3.

---

## 2 Assumptions and Notation

### 2.1 Assumptions

**Assumption 1. The gas in the duct moves as a plug flow with mild mixing.** *Justification:* the duct is long and narrow compared with its diameter, so axial mixing is small relative to transport [10]. We represent the remaining mixing by a first-order lag and test its time constant in Section 8.

**Assumption 2. The flue-gas flow is proportional to the lance oxygen flow.** *Justification:* decarburization consumes most of the blown oxygen during the main blow, so the gas generated per second scales with the oxygen supplied per second. This assumption lets Section 4 measure the duct in oxygen units.

**Assumption 3. The spectrometer has no measurement delay.** *Justification:* the flame is observed optically at the converter mouth, and the optical response is far faster than the 0.5-s sampling step. The flame therefore serves as the real-time reference in Sections 6 and 7.

**Assumption 4. The analyzer reports volume fractions.** *Justification:* Section 3.4 shows that the flow-rate reading violates the carbon–oxygen balance by a factor of about six, whereas the volume-fraction reading agrees with the typical composition of converter gas.

**Assumption 5. Frame-to-frame flame flicker is random variation around a trend.** *Justification:* turbulent combustion at the converter mouth drives rapid flicker that neighbouring frames cannot predict (Section 5.2.2). We therefore treat flicker as a residual rather than as a signal to reproduce.

### 2.2 Notation

> **Table 1. Notation**

| Symbol | Description | Unit |
|---|---|---|
| $k$ | heat index, $k = 1,\dots,10$ | – |
| $I(t,\lambda)$ | flame intensity at time $t$ and wavelength $\lambda$ | counts |
| $\mathbf{X}$ | log-intensity matrix (time × wavelength) | – |
| $\mathbf{L},\ \mathbf{S}$ | low-rank and sparse parts of $\mathbf{X}$ | – |
| $r$ | rank of $\mathbf{L}$ | – |
| $V(t)$ | cumulative oxygen consumption | m³ |
| $Q(t)$ | lance oxygen flow, $Q = \mathrm{d}V/\mathrm{d}t$ | m³/s |
| $y_{\mathrm{CO}}(t),\ y_{\mathrm{CO_2}}(t)$ | CO and CO₂ readings of the analyzer | vol.% |
| $E_{\mathrm{CO}}(t),\ E_{\mathrm{CO_2}}(t)$ | time integrals of the readings | as in data |
| $\Delta V_{\mathrm{pipe}}$ | equivalent duct volume, in oxygen units | m³ |
| $T_{\mathrm{mix}}$ | mixing time constant | s |
| $\tau(Q)$ | flue-gas delay at lance flow $Q$ | s |
| $t_{\mathrm{arr}}(t)$ | time at which gas generated at $t$ reaches the analyzer | s |
| $\hat{y}(t)$ | real-time prediction of the delay-aligned reading | vol.% |
| $N_{\mathrm{p}}$ | probe sampling interval | s |

---

## 3 Data Preprocessing and Consistency Diagnosis

Each attachment contains about 1,400 spectra of 2,047 wavelengths and the corresponding oxygen and gas records. Every model in this paper depends on the time axis, the lance flow and the meaning of the gas readings, so we examine these three items before modelling. Figure 3 shows the workflow, and Table 2 summarizes the ten heats after preprocessing.

> **[Figure 3]** `paper_figures/out/fig03_preprocessing_workflow.pdf`，宽度 1.0。（**草稿，供胡乘风作底稿**：可直接使用，或据此用 TikZ/draw.io 重画；可编辑源文件 `paper_figures/fig03_preprocessing.py`）
> Caption: **Data preprocessing workflow.** Four checks fix the time axis, the lance flow and the meaning of the gas readings.

> **Table 2. Overview of the ten heats after preprocessing**

| Heat | ID | Group | Frames | Step (s) | Duration (min) | Total O₂ (m³) | Saturated pixels (%) |
|---|---|---|---|---|---|---|---|
| 1 | 6B03205 | 6B | 1479 | 0.5 | 12.3 | 3394 | 0.00 |
| 2 | 6B03206 | 6B | 1311 | 0.5 | 10.9 | 3321 | 0.00 |
| 3 | 6B03270 | 6B | 1413 | 0.5 | 11.8 | 3331 | 0.00 |
| 4 | 6B03271 | 6B | 1420 | 0.5 | 11.8 | 3427 | 0.00 |
| 5 | 6C02057 | 6C | 775 | 1.0 | 12.9 | 3540 | 0.27 |
| 6 | 6C02058 | 6C | 1343 | 0.5 | 11.2 | 3340 | 0.22 |
| 7 | 6C02059 | 6C | 1555 | 0.5 | 13.0 | 3601 | 0.00 |
| 8 | 6C02060 | 6C | 1373 | 0.5 | 11.4 | 3307 | 0.01 |
| 9 | 6C02107 | 6C | 1459 | 0.5 | 12.2 | 3406 | 0.07 |
| 10 | 6C02108 | 6C | 1461 | 0.5 | 12.2 | 3489 | 0.08 |

### 3.1 Two Converter Groups and a Unified Wavelength Grid

Heats 1–4 (prefix 6B) cover 340.535–1027.963 nm, whereas Heats 5–10 (prefix 6C) cover 339.390–1023.303 nm. Both grids have 2,047 points. We interpolate every spectrum onto a common grid from 341 to 1023 nm with a 1-nm step. Pixels that reach the 16-bit limit of 65,535 counts are saturated; their true values are only known to exceed this limit, so we exclude them from every fit rather than use them as measurements. Saturation occurs mainly in the 6C group and affects at most 0.27% of the pixels of a heat. The different calibrations indicate two converters or two instruments, which we use later as a natural grouping of the heats.

### 3.2 Corrected Attachment 1 and the Time Step of Heat 5

The original Attachment 1 contains zeros in both gas columns. We use the corrected file, whose spectra and oxygen records match the original file value for value. Heat 5 has only 775 records, although it consumes 3,540 m³ of oxygen, as much as the other heats. Three observations indicate that its step is 1 s rather than 0.5 s. First, a 1-s step yields a blowing time of 12.9 min, well within the 10.9–13.0 min range of the other heats. Second, its oxygen increment per record is twice the typical value. Third, its counter-beat spacing (Section 3.3) is 3 s instead of 11 s. We adopt the 1-s step and evaluate the 0.5-s alternative in the sensitivity analysis.

### 3.3 Beat Artifact of the Oxygen Counter

The cumulative oxygen does not grow smoothly. Its counter advances in steps of about 4.73 m³, and the median step varies by at most 0.03 m³ among the ten heats. Updates arrive roughly once per second, while the logger samples every 0.5 s. The two clocks drift against each other, so the counter skips one update about every 11 s and then reports a double step (Figure 4a). Direct differencing therefore turns a nearly constant flow into a periodic spike train between 0 and twice the true value.

We first apply isotonic regression to keep $V(t)$ non-decreasing. We then estimate $Q(t)$ as the slope of a linear regression over a sliding window. A window of 24 s spans about two beat periods, so the skipped and doubled steps cancel within it. Figure 4b compares three window lengths in the main blow of Heat 3. The 5-s slope still oscillates between 4 and 6 m³/s, and the 12-s slope keeps a visible ripple. With the 24-s window, the residual standard deviation of the flow falls from 0.27 to 0.16 m³/s in Heat 3, and it falls by 24–40% in the other heats. A longer window removes more ripple but also blurs real changes of the lance flow, so 24 s is the shortest window that covers two full beats.

> **[Figure 4]** `paper_figures/out/fig04_o2_counter_beat.pdf`，宽度 1.0。（欧拉奖风格重画；图注已同步）
> Caption: **Beat artifact of the oxygen counter (Heat 3).** (a) Counter increments in a 60-s window of the main blow: single steps of 4.72 m³ and double steps about every 11.5 s. (b) Lance flow estimated by regression slopes over 5-s, 12-s and 24-s windows; SD denotes the residual standard deviation after detrending.

### 3.4 Carbon–Oxygen Balance Check

The problem describes the CO and CO₂ values as volume flow rates in m³/s. We test this reading with a carbon–oxygen balance. Producing one volume of CO consumes half a volume of O₂, and producing one volume of CO₂ consumes one volume of O₂. If the values were flow rates, the oxygen needed by the measured gas would be $0.5E_{\mathrm{CO}} + E_{\mathrm{CO_2}}$. Figure 5 shows that this amount equals 5.46–6.53 times the oxygen actually supplied, with a mean of 5.92 and a coefficient of variation of 6.5%.

The ratio cannot exceed one if the values are flow rates. During the main blow, however, $y_{\mathrm{CO}} + y_{\mathrm{CO_2}}$ stays between 49 and 61 and remains at or below 72, which matches typical converter off-gas composition in percent by volume. We therefore interpret the readings as volume fractions. The cumulative emissions required by Task 2 are still computed as time integrals of the readings, and our later conclusions rely on relative changes rather than on absolute gas volumes. If all blown oxygen leaves the bath as CO and CO₂, the stable ratio across heats implies an effective diluted flue-gas flow of about 16.9 m³/s. Its narrow spread also indicates that the measurement chain behaves consistently across heats, supporting a common delay model.

> **[Figure 5]** `paper_figures/out/fig05_carbon_oxygen_balance.pdf`，宽度 0.7。（欧拉奖风格重画；图注已同步）
> Caption: **Carbon–oxygen balance under the flow-rate reading.** The oxygen required by the measured CO and CO₂ exceeds the supplied oxygen by a factor of 5.5–6.5 in every heat, so the readings are volume fractions.

---

## 4 Delay-Aware Dual-Modal Framework

The three tasks share one question: at what time does the analyzer report the gas that the bath produces now? We answer it with an arrival-time map. Let $V(t)$ be the cumulative oxygen. Under Assumption 1, a parcel of gas generated at time $t$ reaches the analyzer when the gas generated after it has filled the duct. Under Assumption 2, the gas flow is proportional to the lance flow, so this condition can be written in oxygen units:

$$
\int_{t}^{t_{\mathrm{arr}}(t)} Q(s)\,\mathrm{d}s = V\bigl(t_{\mathrm{arr}}(t)\bigr) - V(t) = \Delta V_{\mathrm{pipe}}. \tag{1}
$$

Equation (1) uses cumulative oxygen as the physical clock of the duct and holds under time-varying flow. For a mean transit flow $\bar{Q}$, the plug-flow transport time equals $\Delta V_{\mathrm{pipe}}/\bar{Q}$, so $t_{\mathrm{arr}}(t)$ marks the earliest arrival of a gas parcel before axial dispersion. Adding a first-order mixing stage with time constant $T_{\mathrm{mix}}$ introduces an extra mean lag of $T_{\mathrm{mix}}$, giving the total delay at steady flow $Q$:

$$
\tau(Q) = T_{\mathrm{mix}} + \frac{\Delta V_{\mathrm{pipe}}}{Q}. \tag{2}
$$

Equation (2) casts the delay–flow relation of Task 2 in a directly testable form: total delay decreases as flow rises, and its excess over $T_{\mathrm{mix}}$ scales inversely with $Q$. This transport map also anchors the other two tasks. It defines the real-time prediction target as $y\bigl(t_{\mathrm{arr}}(t)\bigr)$ and tells the online corrector which past prediction a new probe reading verifies. Finally, it establishes cumulative oxygen as a natural coordinate for cross-heat comparison: two heats with equal oxygen consumption have reached the same blowing stage even when their flow profiles differ (Figure 6).

> **[Figure 6]** `paper_figures/out/fig06_arrival_time_map.pdf`，宽度 0.8。（欧拉奖风格重画；图注已同步）
> Caption: **Arrival-time map.** The duct must be refilled by $\Delta V_{\mathrm{pipe}}$ before the gas generated at $t$ reaches the analyzer, so a larger lance flow gives a shorter delay.

---

## 5 Task 1: Spectral Outlier Removal and Continuous Reconstruction

Task 1 asks for two things: the removal of outliers and a continuous surface that can be drawn in three dimensions. The data make both harder than they appear. A heat contains about 2.9 million intensity values, the flame flickers strongly from frame to frame, a few pixels saturate, and two narrow emission lines vary on their own. Filters that work along time only treat the flicker as outliers and ignore the strong correlation between neighbouring wavelengths. Ordinary principal component analysis uses this correlation but is distorted by the outliers it should detect. We therefore separate a low-rank signal from sparse outliers explicitly and build the continuous surface from the low-rank part.

### 5.1 Rank-Constrained Robust Low-Rank Model (RLR)

A flame spectrum is a smooth thermal continuum with a few emission lines. As the blow proceeds, the temperature and the line intensities change, but the spectral shapes stay within a small family. The log-intensity matrix of a heat is therefore nearly low-rank, while spikes and corrupted frames are sparse. RLR uses this structure in four steps.

**Step 1: Log transform and saturation mask.** We set $X_{t\lambda} = \ln I(t,\lambda)$ on the unified grid and exclude saturated pixels from the observed set; the low-rank completion of Step 4 fills them. The log transform converts multiplicative flicker into an additive shift that a single component can absorb.

**Step 2: Low-rank plus sparse decomposition.** We decompose

$$
\mathbf{X} = \mathbf{L} + \mathbf{S} + \mathbf{N}, \qquad \operatorname{rank}(\mathbf{L}) \le r, \tag{3}
$$

where $\mathbf{S}$ captures sparse outliers and $\mathbf{N}$ denotes small background noise. Classical principal component pursuit [1] fails to separate the noise here, yielding ranks of 426–446 in every heat. We therefore fix the rank and alternate two projections [2]. Given $\mathbf{S}$, we set $\mathbf{L}$ to the rank-$r$ truncated SVD approximation of $\mathbf{X}-\mathbf{S}$; given $\mathbf{L}$, we retain in $\mathbf{S}$ the residuals exceeding five times a robust scale. Iteration terminates once the outlier set stabilizes.

**Step 3: Outlier rules.** The robust scale $\sigma_\lambda$ is estimated separately for each wavelength, because the ultraviolet and near-infrared ends carry fewer counts and more noise. A pixel is an outlier if $|X_{t\lambda}-L_{t\lambda}| > 5\sigma_\lambda$. A frame is anomalous if its mean standardized residual has a robust z-score above 5. Within such a frame, we remove only the pixels with $|z|>2.5$ and keep the rest, so that the frame still determines its own low-rank scores. A run longer than 5 s is treated as a change of operation, such as the end of the blow, rather than as an outlier.

**Step 4: Continuous surface.** After removing outliers, we complete the matrix with rank $r$ and write

$$
\hat{I}(t,\lambda) = \exp\Bigl[\mu(\lambda) + \sum_{j=1}^{r} a_j(t)\,\varphi_j(\lambda)\Bigr], \tag{4}
$$

where $\mu$ is the mean log spectrum, the loadings $\varphi_j$ are interpolated by cubic splines and the scores $a_j$ by smoothing splines. Equation (4) can be evaluated at any time and wavelength, and it stays positive by construction. The smoothing parameter of $a_j$ is selected by generalized cross-validation for the reconstruction. The three-dimensional figures use a stronger smoothing ($\lambda_s = 10^4$) to display the trend of the blow.

### 5.2 Model Solution and Results

#### 5.2.1 Low-Rank Structure and Outlier Detection

The first three components explain at least 99.6% of the log-spectral variance in every heat, averaging 99.8%. We select the rank by hiding 10% of the valid pixels on two representative heats. The hold-out error drops from 0.099 at $r=1$ to 0.032 at $r=3$ and rises for larger ranks as additional components fit pixel noise. We therefore set $r=3$ for single-heat denoising. RLR flags 0.65–3.6% of the pixels and 33–117 isolated frames per heat.

The components have a clear physical meaning. In every heat, the score of the first component follows the logarithm of the total intensity with $|r| = 1.00$, so this component measures the overall brightness of the flame. By contrast, the second score is almost independent of brightness ($|r| = 0.03$). Instead, it correlates with the intensity ratio between 800 and 500 nm ($|r| = 0.52$ on average) and with the contrast of the potassium line ($|r| = 0.48$). This component therefore describes the colour of the flame and the strength of its emission lines, two quantities related to the flame temperature.

Figure 7 shows the decomposition for Heat 3. The low-rank part keeps the rise of the flame during ignition, the plateau of the main blow and the decay near the end. The sparse part concentrates on two narrow bands at 767 nm and 589 nm, the two strongest emission peaks of the mean spectrum. These lines fluctuate independently of the continuum and may correspond to the potassium doublet (766.5/769.9 nm) and the sodium D lines.

> **[Figure 7]** `paper_figures/out/fig07_rlr_decomposition.pdf`，宽度 1.0。（欧拉奖风格重画；图注已同步）
> Caption: **Low-rank plus sparse decomposition of Heat 3.** The low-rank part carries the evolution of the blow; the sparse part concentrates on the emission lines near 767 nm and 589 nm.

We test the detector by injecting known outliers into clean spectra of Heats 3 and 8. Spikes of 5–10 robust standard deviations hit 1% of the pixels, and Gaussian bumps of 0.2–0.5 in log intensity distort 1% of the frames. Table 3 compares RLR with a one-dimensional Hampel filter [3]; higher F1 and lower error are better. RLR finds 93% of the spikes with a precision of 87%, while the Hampel filter mistakes flicker for spikes and misses most of them. The Hampel filter also cannot recognize distorted frames, because it judges each frame by its total intensity. Removing the detected outliers lowers the reconstruction error on the corrupted pixels from 0.068 to 0.041.

> **Table 3. Detection of injected outliers (mean of three repetitions on Heats 3 and 8)**

| Detector | Spike precision | Spike recall | Spike F1 | Frame F1 | Error on corrupted pixels |
|---|---|---|---|---|---|
| Hampel filter | 0.255 | 0.190 | 0.218 | 0.000 | 0.073 |
| RLR (ours) | 0.868 | 0.928 | **0.891** | **0.769** | **0.041** |
| No outlier removal | – | – | – | – | 0.068 |

#### 5.2.2 Reconstruction Accuracy

We evaluate reconstruction by hiding 10% of the valid pixels and predicting them from the rest; Table 4 lists the errors, where lower is better. RLR reaches a log error of 0.031, which corresponds to a relative intensity error of 2.0%. Methods that work along time only, such as linear interpolation and Savitzky–Golay filtering, give errors above 0.36. The two-dimensional Whittaker smoother [4] gives 0.157. Using the spectral correlation across wavelengths is therefore the key to accurate completion.

> **Table 4. Hold-out reconstruction error (10% of valid pixels, mean of ten heats and three repetitions)**

| Method | RMSE (log intensity) | Relative error |
|---|---|---|
| Linear interpolation in time | 0.360 | 26.6% |
| Savitzky–Golay filter in time | 0.401 | 30.8% |
| 2-D Whittaker smoother | 0.157 | 10.6% |
| Low-rank completion without outlier removal | 0.033 | 2.2% |
| **RLR (ours)** | **0.031** | **2.0%** |

When whole frames are hidden, every method reaches an error of about 0.36 (RLR: 0.362; linear interpolation: 0.364). This floor reflects the frame-to-frame flicker of the flame, whose standard deviation around the trend surface is 0.30–0.71 in log intensity. The flicker comes from turbulent combustion at the mouth and cannot be inferred from neighbouring frames, so Equation (4) represents it as the residual around a continuous trend. Figure 8 shows a detected anomalous frame and the reconstruction at 600 nm.

> **[Figure 8]** `paper_figures/out/fig08_outlier_reconstruction.pdf`，宽度 1.0。（欧拉奖风格重画；图注已同步）
> Caption: **Outlier detection and reconstruction (Heat 9).** (a) Intensity at 600 nm with the flagged outliers, the continuous reconstruction and the trend surface; the dotted line marks the frame shown in (b). (b) An anomalous bright frame at 9.14 min, partly saturated, rebuilt by RLR from its retained pixels.

#### 5.2.3 Continuous 3D Spectral Surfaces

Figure 9 presents the continuous spectral surfaces of the ten heats. All heats share the same pattern. A dark ignition stage of 1–2 min ends with a sharp rise as the main blow starts. A broad plateau with its maximum between 750 and 900 nm follows, and the intensity declines at the end of the blow. The 6C heats reach higher intensities than the 6B heats and show more saturation, which agrees with the different instrument settings identified in Section 3.1. The timing of the rise and the decline matches the stage breakpoints found from the gas data in Section 6.2, which shows that the flame and the gas describe the same process.

> **[Figure 9]** `paper_figures/out/fig09_spectral_surfaces.pdf`，宽度 1.0。（欧拉奖风格重画；图注已同步）
> Caption: **Continuous spectral surfaces of the ten heats (trend surfaces, log₁₀ intensity).** Every heat shows ignition, a main-blow plateau and an end-of-blow decline; the 6C heats are brighter.

> 数字出处（不进论文）：A 版 `q1_outlier_summary.csv`、`q1_rank_selection.csv`、`q1_injection_test.csv`、`q1_reconstruction_error_summary.csv`、`q1_frame_holdout_summary.csv`、`q1_emission_peaks.csv`；成分物理含义为本轮由 `data_clean/heat*_spec.npz` 计算（第 1 成分得分与 log 总光强 |r| = 1.00；第 2 成分与亮度 |r| = 0.03、与 800/500 nm 比 0.52、与钾线对比度 0.48，均为十炉平均）。

---

## 6 Task 2: Oxygen–Gas Relationship and Variable Flue-Gas Delay

Task 2 contains three questions of increasing depth: how to recover continuous process curves, how the gas depends on the oxygen consumption, and how the delay depends on the lance flow. The first two describe the gas as a function of the blow; the third needs a model of the duct. We answer them in this order, because the stage structure found in the second step also locates the events used to check the delay.

### 6.1 Continuous Reconstruction of the Process Signals

The lance flow $Q(t)$ comes from the 24-s regression slope of Section 3.3, and $V(t)$ is the monotone cubic interpolation of the counter updates. For the gas readings, we remove two kinds of outliers. Stuck readings are runs that stay constant for more than 10 s during the active blow; we find five such runs of 22–94 points. Spikes are detected by a Hampel filter applied to the sequence of value updates, with a floor equal to the measurement noise, which flags at most four points per heat. The cleaned readings are reconstructed by monotone cubic interpolation followed by a smoothing spline. Hiding 10% of the updates gives a reconstruction error of 0.20 for CO (mean 31.0) and 0.10 for CO₂ (mean 13.8), both below 1%. The cumulative emissions are the trapezoidal integrals of the reconstructed readings.

Figure 10 shows the six curves of Heat 3. The flow holds at about 2.4 m³/s during ignition, steps up to about 4.8 m³/s in the main blow and rises again near the end. CO₂ peaks within the first three minutes, whereas CO rises later and stays high until the end of the blow. The cumulative emissions therefore grow in opposite orders: the CO₂ curve rises early and flattens, and the CO curve starts slowly and steepens.

> **[Figure 10]** `paper_figures/out/fig10_process_curves_heat03.pdf`，宽度 1.0。（欧拉奖风格重画；图注已同步）
> Caption: **Continuous process curves of Heat 3:** (a) lance flow, (b) cumulative oxygen, (c, d) CO₂ and CO readings in vol.%, and (e, f) their cumulative emissions in 10³ vol.%·s. CO₂ peaks early and CO late; shading marks the four blowing stages I–IV.

### 6.2 Four-Stage Relationship between Oxygen and Off-Gas

We describe the gas readings as functions of the cumulative oxygen $V$ rather than of time, because $V$ measures how far the blow has progressed. The rank correlations show why a single global trend is insufficient. Averaged over the ten heats, CO correlates with $V$ at $\rho = 0.74$, whereas CO₂ correlates at $\rho = -0.13$. After a cubic time trend is removed, the CO correlation drops to 0.38, so much of the relation is a stage effect rather than a smooth proportionality. The cumulative emissions correlate with $V$ at $\rho \approx 1.00$, but only because all three quantities increase with time.

For each heat, we therefore fit $y_{\mathrm{CO}}(V)$ with a continuous piecewise-linear function with three breakpoints. A grid search locates the breakpoints, with the second one before 60% and the third one after 75% of the total oxygen. The fits reach $R^2 = 0.956$–$0.997$. Table 5 lists the results.

> **Table 5. Four blowing stages identified from CO versus cumulative oxygen (ten heats)**

| Stage | Range of $V/V_{\mathrm{end}}$ | CO share in CO+CO₂ | Mean slope of $y_{\mathrm{CO}}$ (per m³) |
|---|---|---|---|
| I. Early blow (CO₂-dominated) | 0 – 0.23 ± 0.02 | 9.5–28.9% | 0.012 |
| II. Decarburization ramp-up | – 0.43 ± 0.06 | 51.0–61.3% | 0.055 |
| III. Steady decarburization | – 0.90 ± 0.08 | 72.7–80.6% | −0.002 |
| IV. End-of-blow decline | – 1 | 57.8–84.7% | −0.082 |

The four stages follow converter bath chemistry. During Stage I, silicon and manganese consume much of the blown oxygen, and carbon oxidized at low bath temperature leaves primarily as CO₂. As the bath heats up in Stage II, decarburization accelerates and CO rises by 0.055 percentage points per cubic metre of oxygen. Stage III forms a steady plateau where CO remains nearly flat and accounts for roughly three quarters of the carbon-bearing off-gas. Finally, carbon depletion in Stage IV causes CO to decline sharply. The cumulative emissions follow the same breakpoints: a piecewise-linear fit of $E_{\mathrm{CO}}(V)$ and $E_{\mathrm{CO_2}}(V)$ reaches $R^2 = 0.998$ with a lower AIC than logistic or quadratic fits. Between the two converter groups, only the Stage III slope differs significantly (Mann–Whitney test, $p = 0.02$).

> **[Figure 11]** `paper_figures/out/fig11_four_stage_relation.pdf`，宽度 1.0。（欧拉奖风格重画；图注已同步）
> Caption: **CO and CO₂ versus cumulative oxygen for the ten heats.** Three breakpoints (dots) divide every blow into the same four stages (shading); dashed lines are the piecewise-linear fits of CO.

### 6.3 Plug-Flow Delay Model (PFD)

To estimate $\Delta V_{\mathrm{pipe}}$ in Equation (1), we need a signal whose shape is common to all heats, so that the shift between heats and their own oxygen clock can be identified. We use the oxygen-equivalent gas signal $y_{\mathrm{ox}} = y_{\mathrm{CO_2}} + 0.5\,y_{\mathrm{CO}}$, which measures the oxygen bound in the carbon gas. By the balance of Section 3.4, its time integral is proportional to the oxygen supplied, with a ratio that varies by only 6.5% between heats. In the oxygen coordinate, the ten heats indeed share one trajectory (Figure 12b). PFD describes this trajectory in three steps.

**Step 1: Shifted generation curve.** The gas reaching the analyzer at cumulative oxygen $V$ was generated at $V^{+} = \max(V-\Delta V_{\mathrm{pipe}},\,0)$. We write the generation curve as

$$
u(V) = g(V)\,\bigl[\eta + (1-\eta)\,s\bigl((V^{+}-V_{\mathrm{m}})/w_{\mathrm{m}}\bigr)\bigr]\; s\bigl(-(V^{+}-V_{\mathrm{e}})/w_{\mathrm{e}}\bigr), \tag{5}
$$

where $s(\cdot)$ is the logistic function and $g(V) = s\bigl((V-\Delta V_{\mathrm{pipe}})/3.5\bigr)$ switches the signal on when the first gas arrives. The parameter $\eta$ sets the ignition level, $(V_{\mathrm{m}}, w_{\mathrm{m}})$ locates the main-blow rise, and $V_{\mathrm{e}}$ locates the end-of-blow decline with a transition width $w_{\mathrm{e}}$ fixed at 180 m³.

**Step 2: Mixing and scaling.** A first-order filter with time constant $T_{\mathrm{mix}}$ converts $u$ into the modelled reading, and a heat-specific factor scales it to the data.

**Step 3: Estimation.** We minimize the mean of $1-R^2$ over the heats with L-BFGS-B [5] from three starting points far from each other. Because the five parameters differ by four orders of magnitude, we rescale them to unit size before optimization. We fix the mixing constant $T_{\mathrm{mix}}$ at 5 s, because fits with 2, 5 and 10 s reach the same $R^2$ (0.9754–0.9760). The uncertainty of $\Delta V_{\mathrm{pipe}}$ comes from 250 bootstrap resamples of whole heats [6], each refitting all five parameters. A synthetic test checks the estimator: data generated with $\Delta V_{\mathrm{pipe}} = 135$ m³ and noise are fitted from blind starting points of 45 and 210 m³, and the estimate returns 135.3 m³.

### 6.4 Model Solution and Results

#### 6.4.1 Duct Volume and the Delay Curve

PFD gives $\Delta V_{\mathrm{pipe}} = 127.1$ m³ with a 95% bootstrap interval of 123.4–132.6 m³ and a global $R^2$ of 0.976. The estimate is stable: calibrating it ten times, each time without one heat, gives 127.1 ± 0.9 m³. Substituting it into Equation (2) gives

$$
\tau(Q) = 5 + \frac{127.1}{Q}\ \text{s}. \tag{6}
$$

Table 6 evaluates Equation (6) at typical flows. At the ignition flow of 2.35 m³/s, the gas needs 59 s to reach the analyzer; in the main blow at 5 m³/s, it needs 30 s. Doubling the flow from 3 to 6 m³/s shortens the delay by 21 s. Fitting the 6B and 6C groups separately changes the transport time by at most 0.2 s.

> **Table 6. Flue-gas delay at different lance flow rates (PFD, 95% bootstrap interval)**

| $Q$ (m³/s) | 2.35 | 3.0 | 4.0 | 4.75 | 5.0 | 5.5 | 6.0 | 7.0 |
|---|---|---|---|---|---|---|---|---|
| $\tau$ (s) | 59.1 | 47.4 | 36.8 | 31.8 | 30.4 | 28.1 | 26.2 | 23.2 |
| 95% interval (s) | 57.5–61.4 | 46.1–49.2 | 35.8–38.2 | 31.0–32.9 | 29.7–31.5 | 27.4–29.1 | 25.6–27.1 | 22.6–23.9 |

> **[Figure 12]** `paper_figures/out/fig12_plug_flow_delay.pdf`，宽度 1.0。（欧拉奖风格重画；图注已同步）
> Caption: **Plug-flow delay model.** (a) $\tau(Q)$ with the 95% bootstrap band and the ignition and main-blow onset events. (b) Oxygen-equivalent readings of the ten heats collapse onto one trajectory in the oxygen coordinate. (c) Recalibrating $\Delta V_{\mathrm{pipe}}$ without each heat gives 127.1 ± 0.9 m³, inside the full-data 95% bootstrap interval (shading).

#### 6.4.2 Validation against Main-Blow Onset Events

The value of $\Delta V_{\mathrm{pipe}}$ is anchored mainly by the arrival of the first gas after ignition. Because ignition itself takes some time, we check the model against the main blow (Table 7). When the lance flow first exceeds 3.5 m³/s, the oxygen-equivalent signal crosses 13 vol.% after 28–45.5 s in the ten heats, with a median of 34.5 s. These event times include the build-up of decarburization, so they bound the transport delay from above. For the mean flows of 4.4–6.6 m³/s over these intervals, Equation (6) gives 24–34 s, below the bound as expected. A regression of all twenty ignition and main-blow events on $1/Q$ reaches $R^2 = 0.74$.

#### 6.4.3 Model-Free Check by Windowed Cross-Correlation

This check requires no transport model. A cross-correlation between the first spectral component and CO over 120-s windows yields 44 windows with a distinct peak. Their delay decreases as window-mean flow increases (Spearman $\rho = -0.75$, $p = 3.4\times10^{-9}$), falling from 89 s below 4.5 m³/s (three windows) and 63 s at 4.5–5.0 m³/s to 31 s at 5.0–5.5 m³/s and 14.5 s above 5.5 m³/s (Figure 13). Because window time correlates only weakly with flow ($\rho = -0.12$), this decline is not a stage artifact.

> **Table 7. Delay estimates from independent methods**

| Method | Flow range (m³/s) | Delay (s) |
|---|---|---|
| PFD, Equation (6) | 5.0 | 30.4 (29.7–31.5) |
| Main-blow onset events (10 events, upper bound) | 4.4–6.6 | 28–45.5, median 34.5 |
| Windowed cross-correlation (44 windows) | 3.2–5.7 | median 40.5; decreases with $Q$, $\rho=-0.75$ |
| Constant delay maximizing spectral predictability | full blow | 31 |
| First gas after ignition (upper bound) | 2.35 | median 49.8 |

> **[Figure 13]** `paper_figures/out/fig13_windowed_delay_vs_flow.pdf`，宽度 0.7。（欧拉奖风格重画；图注已同步）
> Caption: **Windowed delay estimates versus lance flow.** Dots are 44 window estimates and squares are the medians of four flow bins with interquartile ranges; the curve is the PFD model of Equation (6).

The windowed estimates fall faster with the flow than $1/Q$ does. This suggests that the duct volume is not the only factor; a higher flow may also enhance mixing and post-combustion in the hood. In the main-blow flow range of 4.4–6.6 m³/s, Equation (6) gives 24–34 s. These values stay below the two event-based upper bounds and agree with the constant delay of 31 s that maximizes spectral predictability. We therefore adopt Equation (6) as the delay model for Task 3 and test the consequence of a ±10% error in $\Delta V_{\mathrm{pipe}}$ in Section 8.

> 数字出处（不进论文）：B 版 `q2_delay_vs_flowrate_table.csv`、`q2_tmix_identifiability_sensitivity.csv`、`q3_per_fold_delay_params.csv`、`q2_step_response_delay_events.csv`、`q2_mass_balance_consistency.csv`、`q2_detrended_residual_corr.csv`、`verification_audit_report.csv`（合成回收 135.29 m³）；A 版 `q2_stages.csv`、`q2_stage_group_test.csv`、`q2_cumulative_fit.csv`、`q2_gas_holdout_error.csv`、`q2_reconstruction_summary.csv`、`q2_delay_auxiliary.csv`、`q2_delay_window_bins.csv`、`q2_delay_model_comparison.csv`。

---

## 7 Task 3: Real-Time Off-Gas Prediction from Flame Spectra

Task 3 combines the low-rank spectral structure of Section 5 with the delay model of Section 6. It differs from an ordinary regression in four ways. The target is delayed, so the flame at time $t$ must be matched with the gas that the analyzer reports $\tau$ seconds later. Only ten heats are available, and a model learned on one heat fails on another. The problem requires the heat-wise models to be combined. Finally, a real-time model may use only past data, and its evaluation must not let the test heat leak into any tuning step. Our soft sensor addresses these four points in turn.

### 7.1 Prediction Target and Causal Features

At time $t$, the flame reflects the bath state at $t$, whereas the analyzer reports that gas only after plug-flow transit at $t_{\mathrm{arr}}(t)$. We therefore define the real-time target as the delay-aligned reading $y\bigl(t_{\mathrm{arr}}(t)\bigr)$ from Equation (1), providing a prediction roughly 25–30 s before the analyzer measurement arrives.

All features use measurements up to time $t$ only. For each frame, we subtract the ultraviolet dark baseline below 385 nm, smooth along wavelength and resample onto 127 points between 380 and 1010 nm to trim noisy spectral edges. RLR serves the offline analysis of Task 1, where saturated pixels are simply filled. An online predictor should also respect that a saturated pixel lies above the detector limit. We therefore clean the training spectra with a censoring-aware variant of the same low-rank idea. It applies a soft-thresholded rank-12 SVD to the log spectra, keeps every filled saturated pixel at or above the saturation level and then smooths along wavelength (Appendix B). We pool the cleaned spectra of all training heats and both converters and extract an eight-dimensional basis. It is wider than the three components of a single heat because it must also cover the differences between heats and converters. Projecting each new frame onto this basis (excluding saturated pixels) compresses 127 correlated channels into eight scores.

Three physical descriptors complement the scores. We derive a colour temperature from the Wien approximation: across three line-free continuum windows (480–575, 610–750 and 785–960 nm), the slope of $\ln(I\lambda^{5})$ versus $1/\lambda$ equals $-c_2/T_{\mathrm{c}}$ with $c_2 = 14{,}388$ µm·K. Spectral centroid and spectral entropy further quantify where and how broadly the radiation distributes. Causal exponential moving averages at 5, 25, 80 and 200 s capture both flicker-suppressed instantaneous levels and stage-scale trends, while causal process features supply $V(t)$, lance flow and blowing progress.

### 7.2 Dual-Modal Soft Sensor (DSS)

**Step 1: Linear-tree predictor.** A ridge regression on bounded linear features captures global trends and supports extrapolation beyond the training range. This matters for Heat 1, whose CO reading is the highest of all heats between 30% and 65% of the total oxygen and peaks at 59 near the middle of the blow. Gradient-boosted trees [7] then model the residual nonlinearity.

**Step 2: Model combination.** Models trained on a single heat fail to transfer: across all off-diagonal pairs, the median R² is −0.02 and 52% of pairs are negative. We therefore combine training heats rather than selecting one. Three schemes address the combination requirement: pooling all training heats, hard grouping by converter or cluster, and soft-gated weighting by similarity to the test heat. For the soft-gated scheme, let $\mathbf{d}_j$ be a fingerprint of heat $j$ computed from its first 60 s: the mean normalized spectrum, mean Task-1 scores, mean flow and mean log total intensity. Training heat $j$ receives sample weight

$$
w_j = 0.55 + 0.45\,M\,\pi_j, \qquad \pi_j \propto \exp\Bigl(-0.65\,\frac{\lVert\mathbf{d}_j-\mathbf{d}_{\mathrm{test}}\rVert^2}{\operatorname{med}_i\lVert\mathbf{d}_i-\mathbf{d}_{\mathrm{test}}\rVert^2} + 0.55\,\mathbb{1}_{\text{same group}}\Bigr), \tag{7}
$$

where $M$ is the number of training heats and $\sum_j \pi_j = 1$. The uniform term retains information from every heat, while the similarity term emphasizes heats with comparable ignition profiles. The fingerprint uses only the first 60 s of the test heat, so the weights are fixed one minute after ignition, before the main decarburization starts, and use no later data.

**Step 3: Validation protocol.** The outer loop runs leave-one-heat-out (LOHO) validation, so each of the ten heats is predicted by a model trained strictly without it. Every data-driven quantity, including the cross-heat spectral basis and $\Delta V_{\mathrm{pipe}}$, is recalibrated inside each fold. An inner eight-versus-one loop selects the correction gain and conformal interval width, ensuring the test heat does not enter any tuning step. Two automated tests verify this isolation: corrupting future frames of a heat leaves all earlier predictions unchanged, and corrupting the test heat leaves all trained quantities unchanged.

**Step 4: Smith-type correction.** A probe reading arriving at time $t$ verifies the earlier prediction made at $t_{\mathrm{arr}}^{-1}(t)$, following the Smith predictor for dead-time systems [8]. The verified error updates a smoothed bias term that corrects the current prediction:

$$
b_k = \alpha\,e_k + (1-\alpha)\,b_{k-1}, \qquad \hat{y}_{\mathrm{c}}(t) = \hat{y}(t) + g\,b_k, \tag{8}
$$

where $k$ indexes the latest probe arrival up to time $t$, $e_k$ is its verified error, $\alpha$ is a smoothing factor and $g$ is the inner-loop gain. Under probe interval $N_{\mathrm{p}}$, the bias updates only when a new reading arrives.

**Step 5: Prediction interval.** Split conformal prediction [9] scales inner-loop residuals by a local spread to construct a 90% prediction interval.

### 7.3 Model Solution and Results

#### 7.3.1 Ablation and Model Combination

Table 8 compares the variants under LOHO, where higher R² and lower RMSE indicate stronger accuracy. Delay alignment is essential: without it, spectra plus process features predict delay-aligned CO₂ at a mean R² of only 0.33; aligning targets raises R² to 0.66. Pooling and soft-gated weighting achieve comparable accuracy (CO: 0.881 versus 0.873), whereas hard grouping degrades accuracy because each group retains only three to five training heats. We adopt the soft-gated model as our zero-probe DSS because it closely matches pooling across all ten heats while providing an explicit heat-combination mechanism and a unified base for Smith correction.

> **Table 8. Leave-one-heat-out performance (mean R² and RMSE over ten heats)**

| Model | Probe readings saved | CO R² | CO RMSE | CO₂ R² | CO₂ RMSE |
|---|---|---|---|---|---|
| Process features only | 100% | 0.860 | 6.46 | 0.669 | 2.46 |
| Spectra + process, unaligned | 100% | 0.801 | 7.65 | 0.327 | 3.41 |
| Spectra + process, aligned (pooled) | 100% | 0.881 | 6.01 | 0.663 | 2.44 |
| Grouped by converter (6B/6C) | 100% | 0.757 | 7.73 | 0.593 | 2.69 |
| **DSS, soft-gated (zero probe)** | 100% | **0.873** | **6.18** | **0.658** | **2.45** |
| DSS + Smith, $N_{\mathrm{p}}$ = 120 s | 99.6% | 0.908 | 5.28 | 0.695 | 2.31 |
| DSS + Smith, $N_{\mathrm{p}}$ = 60 s | 99.2% | 0.920 | 4.93 | 0.726 | 2.19 |
| DSS + Smith, continuous | 0% | **0.949** | **3.85** | **0.819** | **1.74** |
| Lagged analyzer reading, continuous | 0% | 0.916 | 4.87 | 0.538 | 2.86 |

The process features alone already explain much of the CO variation, because the blowing progress determines the stage. The spectra add information within each stage. Near the stage breakpoints, the DSS lowers the CO error from 8.02 to 7.55 and improves the steady stages for both gases. In the 10% of time with the fastest flow changes, its error exceeds that of the process-only model. The flame responds to a flow change at once, whereas the gas follows with a lag.

#### 7.3.2 Generalization to All Ten Heats

Figure 14 presents the LOHO predictions across all ten heats. The zero-probe DSS tracks ignition, decarburization ramp-up and end-of-blow decline in every heat. Its CO R² exceeds 0.85 in seven heats and spans 0.67 to 0.97 overall. CO₂ accuracy is lower in Heats 5 (0.33), 1 (0.39) and 9 (0.50). Heat 1 has a higher CO reading than every other heat in the first half of the main blow, while Heat 5 combines coarser 1-s sampling with the highest saturation rate.

The conformal intervals perform as designed. Without probe readings, the 90% intervals average a half-width of 10.1 for CO and cover 87.0% of readings. Coverage drops to 62% in the two heats with the highest CO readings (Heats 1 and 3) and reaches 76–100% in the remaining heats. Under continuous correction, the half-width narrows to 5.7, and coverage reaches 89.8% for CO and 89.75% for CO₂, close to the nominal target level.

> **[Figure 14]** `paper_figures/out/fig14_loho_predictions.pdf`，宽度 1.0。（欧拉奖风格重画；图注已同步）
> Caption: **Leave-one-heat-out predictions of CO and CO₂ for the ten heats.** Dashed lines are the zero-probe DSS predictions, bands their 90% conformal intervals, and solid lines the delay-aligned readings. The model tracks every blowing stage; Heat 1 (shaded), whose CO in the first half of the main blow exceeds that of all other heats, is the hardest extrapolation case.

#### 7.3.3 Reducing Probe Usage

The DSS shifts the probe from a continuous sensor to an occasional calibrator. Figure 15b plots accuracy against probe interval. With one reading every 120 s, corrected predictions reach CO R² = 0.908 and CO₂ R² = 0.695, whereas holding the lagged reading yields only 0.593 and −1.04. Even under continuous sampling, the corrected prediction outperforms the lagged reading (0.949 versus 0.916 for CO) by compensating for the 30-s transport delay.

This trade-off yields clear operating points. Sampling every 60 s saves 99.2% of probe readings while maintaining CO R² at 0.920; sampling every 30 s saves 98.3% and reaches 0.934. In a partial least squares analysis, the highest-importance wavelengths lie at 765–770 nm and 590 nm, the potassium and sodium lines isolated in Task 1, followed by the 730–825 nm continuum. The same atomic emission lines thus separate sparse outliers in Task 1 and carry off-gas information in Task 3.

> **[Figure 15]** `paper_figures/out/fig15_transfer_probe_vip.pdf`，宽度 1.0。（欧拉奖风格重画；图注已同步）
> Caption: **Cross-heat transfer, probe-saving trade-off and wavelength importance.** (a) Single-heat transfer matrix (diagonal greyed out) showing poor one-to-one generalization. (b) Mean LOHO R² versus probe interval; intervals of 120, 60, 30 and 15 s save 99.6%, 99.2%, 98.3% and 96.7% of probe readings. With one reading every 120 s, Smith correction keeps CO R² above 0.9, whereas the lagged reading falls to 0.59. (c) Potassium and sodium lines carry the highest PLS VIP scores.

> 数字出处（不进论文）：B 版 `q3_loho_ablation_summary.csv`、`q3_loho_per_heat_metrics.csv`（Combined_SoftGated_Hybrid 逐炉 CO：0.673/0.775/0.806/0.939/0.897/0.959/0.856/0.962/0.970/0.891；CO₂ 最弱为 H05 0.333、H01 0.386、H09 0.495）、`q3_single_heat_transfer_matrix.csv`、`q3_conformal_90pct_coverage.csv`（零探头 CO 逐炉 0.62–1.00、均值 0.870、半宽 10.14；连续校正半宽 5.65）、`q3_transition_regime_rmse.csv`、`q3_pls_vip_wavelength.csv`、`verification_audit_report.csv`（第 5、6 项泄漏检验）；颜色温度与指纹定义见 `q3_fusion_predict.py`；Heat 1/Heat 3 的 CO 比较（30%–65% 耗氧区间 Heat 1 最高，70% 之后 Heat 3 最高，峰值 63.8）见 `B_results/tables/q3_co_by_oxygen_fraction.csv`（由原始数据按耗氧分数分箱取中位数）。

---

## 8 Sensitivity Analysis

We perturb the parameters that support the main conclusions and judge each perturbation by the quantity it could change. Table 9 summarizes the results.

> **Table 9. Sensitivity of the main results**

| Task | Parameter | Values tested | Effect |
|---|---|---|---|
| 1 | Outlier threshold | 4σ, 5σ, 6σ | flagged pixels 2.5%, 2.2%, 2.1%; anomalous frames unchanged |
| 1 | Rank $r$ | 2–6 | hold-out error 0.046, 0.032, 0.035, 0.041, 0.040 |
| 2 | Mixing constant $T_{\mathrm{mix}}$ | 2, 5, 10 s | $\Delta V_{\mathrm{pipe}}$ = 131.9, 127.1, 122.0 m³; $\tau$ at 4.5 m³/s = 31.3, 33.3, 37.1 s |
| 2 | Time step of Heat 5 | 1 s, 0.5 s | transport time changes by less than 0.5 s |
| 2 | Converter group | 6B, 6C | transport time differs by at most 0.2 s |
| 3 | $\Delta V_{\mathrm{pipe}}$ | −10%, +10% | zero-probe DSS: R² changes by ≤ 0.003 (CO, CO₂) |
| 3 | $\Delta V_{\mathrm{pipe}}$ with Smith correction | −10%, +10% | CO R² within 0.003; CO₂ R² 0.706 → 0.755 at $N_{\mathrm{p}}$ = 60 s |
| 3 | Probe interval $N_{\mathrm{p}}$ | 15–120 s, continuous | CO R² 0.943–0.908, 0.949 |

For Task 1, a stricter threshold removes fewer pixels but leaves the number of anomalous frames unchanged, and ranks from 3 to 6 give similar hold-out errors. The outlier rules and the rank therefore do not drive the reconstruction.

For Task 2, the mixing constant and the duct volume trade off against each other. The total delay at the main-blow flow still varies by less than 6 s, and the two converter groups share one duct model. The choice of the time step of Heat 5 also has a negligible effect.

The most critical check concerns Task 3, where the delay defines the prediction target. We perturb the duct volume by ±10% during both training and Smith correction while still evaluating against the calibrated target. Zero-probe accuracy shifts by at most 0.003, and corrected CO accuracy likewise stays within 0.003. By contrast, CO₂ correction improves slightly as the assumed volume grows. This aligns with the spectral-predictability criterion, whose constant lag maximizing predictability is 72 s for CO₂ versus 31 s for CO. Both results suggest that the CO₂ reading follows the flame more slowly than the CO reading; the present data do not identify the cause.

> 数字出处（不进论文）：A 版 `q1_threshold_sensitivity.csv`、`q1_rank_selection.csv`、`q2_delay_model_comparison.csv`（CO₂ 最优常数时延 72 s）、`q3_delay_misestimation_sensitivity.csv`（本次补算，脚本 `B_results/code/sens_delay_misestimation.py`）；B 版 `q2_tmix_identifiability_sensitivity.csv`、`q2_delay_vs_flowrate_table.csv`、`q3_loho_ablation_summary.csv`。

---

## 9 Model Evaluation and Further Discussion

### 9.1 Strengths

- **Data-consistency diagnosis.** Before fitting any model, we adopt the corrected Attachment 1, establish the 1-s step of Heat 5, suppress the oxygen-counter beat and show that the gas readings are volume fractions.
- **Unified low-rank structure for cleaning and continuity.** RLR uses the same three components to detect outliers, fill saturated pixels and build continuous surfaces, with each step verified by injection testing.
- **Physically grounded delay model.** PFD derives flow dependence from plug-flow transport. Its main-blow delay agrees with independent constant-delay estimates, stays below event-based upper bounds and matches the negative delay–flow trend from windowed cross-correlation.
- **Strictly causal, leakage-free validation.** Every learned parameter is recalibrated within each LOHO fold, and the test heat does not enter any tuning step.
- **Practical soft sensor.** The DSS forecasts off-gas states about 30 s ahead while keeping CO R² above 0.9 with 99.6% fewer probe readings.

### 9.2 Weaknesses

- **Limited sample of ten heats.** Heats 1 and 3 have the highest CO readings of the ten heats, so the zero-probe model extrapolates when either is held out, and coverage falls to 62%. Additional heats, particularly high-carbon blows, would widen the training envelope.
- **Narrow main-blow flow range.** Median lance flows across heats cluster between 4.6 and 5.1 m³/s, so delay–flow identification rests on modest flow variations. Under Assumption 1, $T_{\mathrm{mix}}$ cannot be separated from $\Delta V_{\mathrm{pipe}}$ without deliberate flow steps.
- **Undocumented gas units.** Assumption 4 relies on the carbon–oxygen balance rather than instrument manuals, leaving absolute flue-gas volume unverified until analyzer calibration records become available.

### 9.3 Further Discussion

Additional heats with deliberate lance-flow steps would separate $T_{\mathrm{mix}}$ from $\Delta V_{\mathrm{pipe}}$ and test the steeper power-law delay suggested by the windowed estimates. Radiometric calibration of the spectrometer would further convert colour temperature into absolute flame temperature. Moreover, pairing predicted CO with the Stage IV decline could provide an early endpoint indicator for decarburization.

Our framework transfers readily to other converters. Equation (1) requires only cumulative oxygen and one calibrated duct volume, and the cross-heat spectral basis can be re-estimated from a handful of heats. The similarity weights in Equation (7) then let each new heat borrow strength from comparable past blows. In plant operation, operators can select a probe interval from Figure 15b to balance probe wear against accuracy, while the conformal interval flags heats moving outside prior training experience.

---

## 10 Conclusions

We developed a delay-aware framework that fuses real-time flame spectra with delayed flue-gas readings in converter steelmaking. Prior data diagnosis resolved Attachment 1, established the time step of Heat 5, eliminated the oxygen-counter beat and identified the gas readings as volume fractions.

For Task 1, our Rank-Constrained Robust Low-Rank model decomposed each log spectrum into three physical components and sparse outliers. It detected injected spikes with an F1 score of 0.89, reconstructed held-out pixels within a 2.0% relative error and generated continuous spectral surfaces for all ten heats.

For Task 2, the reconstructed process signals revealed four blowing stages with breakpoints at 23%, 43% and 90% of total oxygen. The Plug-Flow Delay model yielded $\tau(Q) = 5 + 127.1/Q$ s, which drops from 47.4 s at 3 m³/s to 30.4 s at 5 m³/s, and independent checks confirmed that delay decreases with flow.

For Task 3, the Dual-Modal Soft Sensor combined heat-level models through causal similarity weighting. Without probe readings, it predicted delay-aligned CO and CO₂ at mean R² of 0.873 and 0.658 across all ten heats; adding one probe reading every 120 s raised R² to 0.908 and 0.695. Flame spectra can thus replace most probe readings while delivering off-gas information roughly 30 s earlier.

---

## References

[1] Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust principal component analysis? *Journal of the ACM*, 58(3), Article 11.

[2] Zhou, T., & Tao, D. (2011). GoDec: Randomized low-rank & sparse matrix decomposition in noisy case. In *Proceedings of the 28th International Conference on Machine Learning* (pp. 33–40).

[3] Hampel, F. R. (1974). The influence curve and its role in robust estimation. *Journal of the American Statistical Association*, 69(346), 383–393.

[4] Eilers, P. H. C. (2003). A perfect smoother. *Analytical Chemistry*, 75(14), 3631–3636.

[5] Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing*, 16(5), 1190–1208.

[6] Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

[7] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).

[8] Smith, O. J. M. (1957). Closer control of loops with dead time. *Chemical Engineering Progress*, 53(5), 217–219.

[9] Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.

[10] Levenspiel, O. (1999). *Chemical Reaction Engineering* (3rd ed.). Wiley.

[11] Wang, Z., Liu, Q., Liu, H., & Wei, S. (2020). A review of end-point carbon prediction for BOF steelmaking process. *High Temperature Materials and Processes*, 39(1), 653–662.

[12] Zhao, B., Zhao, J., Wu, W., Zhang, F., & Yao, T. (2023). Research on prediction model of converter temperature and carbon content based on spectral feature extraction. *Scientific Reports*, 13(1), Article 14409.

[13] Kadlec, P., Gabrys, B., & Strandt, S. (2009). Data-driven soft sensors in the process industry. *Computers & Chemical Engineering*, 33(4), 795–814.

---

## Appendix（不计入正文页数）

- A. Software: Python 3.13; NumPy, SciPy, pandas, scikit-learn, XGBoost, LightGBM, Matplotlib.
- B. Source code: `B_results/code/`（Task 1 与数据诊断、Task 2 分阶段与窗口时延、敏感性补算）与 `solution_v4_verified/solution_code/`（Task 2 PFD 与 Task 3 DSS）。
  > 附录放代码前必须清理（排版说明，不进论文）：`solution_v4_verified/solution_code/` 7 个文件共约 270 行中文注释，并含“v4”“红队审计”“P0/P1 修复”等过程字样，APMCM 不允许论文出现中文。需改成英文注释、删去过程字样；附录里的目录名改为中性名称（如 `code_task1/`、`code_task23/`），不出现 v4、verified、frozen。只改注释和名称，不改代码逻辑，改完重跑一次确认结果不变。7.1 节提到的 censoring-aware cleaning 对应 `q1_spectral_recon.py` 的 `reconstruct_single_spectrum` 与 `extract_fold_q1_spectral_basis`，附录 B 中要能找到。
- C. Additional figures: single-heat spectral surfaces (`q1_surface_heat01–10`), single-heat process curves (`q2_curves_heat01–10`).
- D. Report on use of AI（若官方要求）。
