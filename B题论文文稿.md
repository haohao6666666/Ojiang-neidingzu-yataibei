# B 题论文文稿（英文正文，v2：扩充至约 6,300 词，按 O 奖写作规范全文润色）

> 使用说明（这一部分不进入论文）
> - 本文稿只写文字，不做 LaTeX/Word 排版。每处图表用方框标注：**图号、文件路径、建议宽度、图注**。标注“PLACEHOLDER”的图尚未绘制，排版时先放占位框。
> - 所有数字取自冻结结果：A 版 `B_results/tables/`，B 版 `solution_v4_verified/solution_outputs/tables/`（提交 `416495a`），以及本轮补算的 `B_results/tables/q3_delay_misestimation_sensitivity.csv`。每节末尾的“数字出处”只供核对，不进入论文。
> - 篇幅：摘要 466 词；正文（引言至结论）6,321 词，另有图注约 490 词与 9 张表；15 张图。按 APMCM 模板粗估 25–27 页，**有超出 25 页上限的风险**。排版时按顺序处理：① 正文用 11 pt（规范允许 11 或 12 pt）；② 仍超页则把 Table 2、Figure 3、Figure 8 移入附录（正文保留一句引用）；③ 再超则把 Table 9 压缩为 5 行。
> - 参考文献只列已确认真实存在的经典文献；转炉火焰光谱与烟气分析的领域文献需要队员补充并核实后再加入。

---

# Seeing the Off-Gas Before It Arrives: Delay-Aware Fusion of Flame Spectra and Flue-Gas Signals for Converter Steelmaking

## Summary

In converter steelmaking, the flame at the converter mouth reports the bath reactions in real time, whereas the flue-gas analyzer reports them only after the gas has crossed the duct, about half a minute later. We develop a **Delay-Aware Dual-Modal Framework** that cleans both data streams, quantifies the transport delay and reads the off-gas state from the flame before the analyzer measures it.

Before modelling, we diagnose the data. We use the corrected Attachment 1 and identify a sampling interval of 1 s for Heat 5. The cumulative-oxygen counter advances in steps of about 4.73 m³ and adds a double step every 11 s, so we estimate the lance flow from a 24-s regression slope instead of direct differencing. A carbon–oxygen balance shows that, read as flow rates, the CO and CO₂ values would require **5.5–6.5 times** the supplied oxygen; we therefore treat them as volume fractions.

**Task 1: Spectral cleaning and continuity.** We build a **Rank-Constrained Robust Low-Rank model (RLR)** that splits each log spectrum into three components and sparse outliers. The first component tracks the flame brightness and the second its colour. In an injection test, RLR detects spikes with an F1 score of **0.89** (Hampel filter: 0.22), and the continuous surface reproduces held-out pixels with a relative error of **2.0%**. The sparse outliers concentrate on the potassium and sodium lines, which fluctuate independently of the continuum.

**Task 2: Oxygen, gas and delay.** We reconstruct six continuous process curves per heat and identify four blowing stages, with breakpoints at **23%, 43% and 90%** of the total oxygen (piecewise R² = 0.956–0.997). In the oxygen coordinate, the gas trajectories of the ten heats collapse onto one curve, which the **Plug-Flow Delay model (PFD)** fits with R² = 0.976. The model gives **τ(Q) = 5 + 127.1/Q s**, so the delay falls from 47.4 s at 3 m³/s to **30.4 s** at 5 m³/s. A windowed cross-correlation confirms that the delay decreases with the flow (Spearman ρ = −0.75, p = 3.4×10⁻⁹).

**Task 3: Real-time prediction.** We build a **Dual-Modal Soft Sensor (DSS)** that projects each frame onto the RLR basis and aligns the target with the PFD delay. It combines the models of different heats through similarity weighting. Under leave-one-heat-out validation on all ten heats, it predicts CO and CO₂ with mean R² of **0.873** and **0.658** without any probe reading. A Smith-type correction with one probe reading every 120 s, i.e. **99.6% fewer readings**, raises R² to **0.908** and **0.695**, about 30 s ahead of the analyzer.

Sensitivity analysis shows that a ±10% error in the duct volume changes the zero-probe R² by at most 0.003, and that the stage and delay results hold for both converter groups. The flame spectrum can therefore replace most probe readings while delivering the off-gas information earlier.

**Keywords:** robust low-rank decomposition; plug-flow delay; soft sensor; leave-one-heat-out validation; Smith predictor; converter steelmaking

---

## Contents

（排版时自动生成，列到三级标题。）

---

## 1 Introduction

### 1.1 Background

The basic oxygen converter turns hot metal into steel by blowing high-purity oxygen through a lance at supersonic speed. Silicon, manganese, phosphorus and carbon are oxidized in turn, and the carbon leaves the bath as CO and CO₂. Operators must time the end of the blow so that the carbon content and temperature of the steel reach their targets together. They therefore follow the progress of decarburization from two sources. A spectrometer at the converter mouth records the flame radiation every 0.5 s, and a Raman analyzer at the flue-gas bend records the CO and CO₂ contents at the same rate.

The two sources see the same reactions at different times. The flame responds to the bath immediately, while the gas must travel through the hood and the duct before the analyzer can measure it. This transport delay limits endpoint control, because a decision based on the analyzer refers to a bath state that is already half a minute old. The sampling probe also wears quickly in the dusty, high-temperature duct, so frequent sampling is costly. A model that reads the off-gas state from the flame would provide the signal earlier and would allow the probe to work less often.

> **[Figure 1 — PLACEHOLDER]** 自绘示意图：转炉、氧枪、炉口光谱仪（标注 real-time）、烟罩与烟道、弯头处拉曼分析仪（标注 delayed），烟道中画出气团输运箭头，标出 $\Delta V_{\mathrm{pipe}}$。宽度 0.7。
> Caption: **Dual-modal measurement at the converter.** The spectrometer observes the flame in real time; the analyzer observes the same reactions after the gas has crossed the duct volume $\Delta V_{\mathrm{pipe}}$.

### 1.2 Restatement of the Problem

The problem provides the full-process flame spectra (2,047 wavelengths, 0.5-s step), the cumulative oxygen consumption and the CO and CO₂ signals of ten heats. We address the following tasks.

- **Task 1.** Remove the outliers in the spectral data, reconstruct a continuous spectral surface, and draw the three-dimensional spectra of all ten heats.
- **Task 2.** Remove the outliers in the oxygen and gas series, reconstruct continuous curves of the lance flow, cumulative oxygen, CO, CO₂ and their cumulative emissions, relate the gas signals to the oxygen consumption, and compute the flue-gas delay under different lance flow rates.
- **Task 3.** Use the results of Tasks 1 and 2 to predict the flue-gas information in real time from the flame spectra, and test the prediction on all ten heats.
- **Combination requirement.** Combine the models learned from different heats, as a whole or by groups, so that the combined model performs well on all ten heats.

### 1.3 Our Work

Figure 2 summarizes our work. We first diagnose the data and correct four inconsistencies that would otherwise distort every later step. Section 4 then introduces an arrival-time map that links each real-time observation to the delayed gas reading it will produce. The three tasks form a pipeline. Task 1 builds a low-rank model of the spectra and supplies cleaned, continuous spectral features. Task 2 reconstructs the process curves, divides the blow into stages and estimates how the delay depends on the lance flow. Task 3 feeds the spectral features of Task 1 and the delay of Task 2 into a real-time predictor, combines the models of different heats and corrects the prediction whenever a probe reading arrives.

> **[Figure 2 — PLACEHOLDER]** 全宽框架图（三条色带：Data & Diagnosis → Models → Validation & Output）。Data：four data issues；Task 1 RLR → continuous spectral surface；Task 2 signal reconstruction → four stages → PFD delay $\tau(Q)$；Task 3 causal features → DSS → Smith correction → conformal interval；Validation：injection test, hold-out, LOHO, sensitivity。用箭头表示 Task 1 的光谱基和 Task 2 的时延都输入 Task 3。宽度 1.0。
> Caption: **Framework of our work.** The spectral basis of Task 1 and the delay model of Task 2 feed the soft sensor of Task 3.

---

## 2 Assumptions and Notation

### 2.1 Assumptions

**Assumption 1. The gas in the duct moves as a plug flow with mild mixing.** *Justification:* the duct is long and narrow compared with its diameter, so axial mixing is small relative to transport [10]. We represent the remaining mixing by a first-order lag and test its time constant in Section 8.

**Assumption 2. The flue-gas flow is proportional to the lance oxygen flow.** *Justification:* decarburization consumes most of the blown oxygen during the main blow, so the gas generated per second scales with the oxygen supplied per second. This assumption lets Section 4 measure the duct in oxygen units.

**Assumption 3. The spectrometer has no measurement delay.** *Justification:* the flame is observed optically at the converter mouth, and the optical response is far faster than the 0.5-s sampling step. The flame therefore serves as the real-time reference in Sections 6 and 7.

**Assumption 4. The analyzer reports volume fractions.** *Justification:* Section 3.4 shows that the flow-rate reading violates the carbon–oxygen balance by a factor of about six, whereas the volume-fraction reading agrees with the typical composition of converter gas.

**Assumption 5. Frame-to-frame flame flicker is random variation around a trend.** *Justification:* the flicker comes from turbulent combustion at the mouth and cannot be predicted from neighbouring frames (Section 5.2.2), so we model it as a residual rather than as a signal to reproduce.

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

> **[Figure 3 — PLACEHOLDER]** 预处理流程图：Raw attachments → (1) corrected Attachment 1 → (2) time step of Heat 5 → (3) unified wavelength grid → (4) oxygen counter de-beating → (5) carbon–oxygen balance → clean dataset。宽度 0.9。
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

Heats 1–4 (prefix 6B) cover 340.535–1027.963 nm, whereas Heats 5–10 (prefix 6C) cover 339.390–1023.303 nm. Both grids have 2,047 points. We interpolate every spectrum onto a common grid from 341 to 1023 nm with a 1-nm step. Pixels that reach the 16-bit limit of 65,535 counts are saturated; we treat them as right-censored values rather than as measurements. Saturation occurs mainly in the 6C group and affects at most 0.27% of the pixels of a heat. The different calibrations indicate two converters or two instruments, which we use later as a natural grouping of the heats.

### 3.2 Corrected Attachment 1 and the Time Step of Heat 5

The original Attachment 1 contains zeros in both gas columns. We use the corrected file, whose spectra and oxygen records coincide with the original file value for value. Heat 5 has only 775 records, although it consumes 3,540 m³ of oxygen, as much as the other heats. Three observations show that its step is 1 s rather than 0.5 s. First, a 1-s step gives a blowing time of 12.9 min, which falls within the 10.9–13.0 min of the other heats. Second, its oxygen increment per record is twice the typical value. Third, its counter beat (Section 3.3) has a period of 3 s instead of 11 s. We use the 1-s step and report the 0.5-s alternative in the sensitivity analysis.

### 3.3 Beat Artifact of the Oxygen Counter

The cumulative oxygen does not grow smoothly. Its counter advances in steps of about 4.73 m³, and the median step varies by at most 0.03 m³ among the ten heats. Updates arrive roughly once per second, while the logger samples every 0.5 s. The two clocks drift against each other, so the counter skips one update about every 11 s and then reports a double step (Figure 4a). Direct differencing therefore turns a nearly constant flow into a periodic spike train between 0 and twice the true value.

We first apply isotonic regression to keep $V(t)$ non-decreasing. We then estimate $Q(t)$ as the slope of a linear regression over a sliding window. A window of 24 s spans about two beat periods, so the skipped and doubled steps cancel within it. Figure 4b compares three window lengths in the main blow of Heat 3. The 5-s slope still oscillates between 4 and 6 m³/s, and the 12-s slope keeps a visible ripple. With the 24-s window, the residual standard deviation of the flow falls from 0.27 to 0.16 m³/s in Heat 3, and it falls by 24–40% in the other heats. A longer window removes more ripple but also blurs real changes of the lance flow, so 24 s is the shortest window that covers two full beats.

> **[Figure 4]** `B_results/figures/fig_q2_o2_beat_artifact.pdf`，宽度 1.0。
> Caption: **Beat artifact of the oxygen counter (Heat 3).** (a) Counter increments in a 60-s window of the main blow: single steps of 4.72 m³ and double steps about every 11.5 s. (b) Lance flow estimated by regression slopes over 5-s, 12-s and 24-s windows; SD denotes the residual standard deviation after detrending.

### 3.4 Carbon–Oxygen Balance Check

The problem describes the CO and CO₂ values as volume flow rates in m³/s. We test this reading with a carbon–oxygen balance. Producing one volume of CO consumes half a volume of O₂, and producing one volume of CO₂ consumes one volume of O₂. If the values were flow rates, the oxygen needed by the measured gas would be $0.5E_{\mathrm{CO}} + E_{\mathrm{CO_2}}$. Figure 5 shows that this amount equals 5.46–6.53 times the oxygen actually supplied, with a mean of 5.92 and a coefficient of variation of 6.5%.

The ratio cannot exceed one if the values are flow rates. During the main blow, however, $y_{\mathrm{CO}} + y_{\mathrm{CO_2}}$ stays between 49 and 61 and never exceeds 72, which matches the composition of converter gas in percent by volume. We therefore interpret the readings as volume fractions. The cumulative emissions required by Task 2 are still computed as time integrals of the readings, and the later conclusions rely on relative changes rather than on absolute gas volumes. If all blown oxygen leaves the bath as CO and CO₂, the stable ratio across heats implies an effective diluted flue-gas flow of about 16.9 m³/s. The small spread of the ratio also shows that the measurement chain behaves alike in all heats, which supports a common delay model.

> **[Figure 5]** `B_results/figures/q2_mass_balance.pdf`，宽度 0.7。
> Caption: **Carbon–oxygen balance under the flow-rate reading.** The oxygen required by the measured CO and CO₂ exceeds the supplied oxygen by a factor of 5.5–6.5 in every heat, so the readings are volume fractions.

---

## 4 Delay-Aware Dual-Modal Framework

The three tasks share one question: at what time does the analyzer report the gas that the bath produces now? We answer it with an arrival-time map. Let $V(t)$ be the cumulative oxygen. Under Assumption 1, a parcel of gas generated at time $t$ reaches the analyzer when the gas generated after it has filled the duct. Under Assumption 2, the gas flow is proportional to the lance flow, so this condition can be written in oxygen units:

$$
\int_{t}^{t_{\mathrm{arr}}(t)} Q(s)\,\mathrm{d}s = V\bigl(t_{\mathrm{arr}}(t)\bigr) - V(t) = \Delta V_{\mathrm{pipe}}. \tag{1}
$$

Equation (1) uses the cumulative oxygen as the clock of the duct, and it holds for a time-varying flow. If $\bar{Q}$ denotes the mean flow during the transit, the transport time equals $\Delta V_{\mathrm{pipe}}/\bar{Q}$. A first-order mixing stage with time constant $T_{\mathrm{mix}}$ adds a mean lag of $T_{\mathrm{mix}}$, so the total delay at a steady flow $Q$ is

$$
\tau(Q) = T_{\mathrm{mix}} + \frac{\Delta V_{\mathrm{pipe}}}{Q}. \tag{2}
$$

Equation (2) states the relation required by Task 2 in a testable form: the delay decreases with the flow, and its excess over $T_{\mathrm{mix}}$ is inversely proportional to the flow. The map also serves the other two tasks. It defines the target of real-time prediction as $y\bigl(t_{\mathrm{arr}}(t)\bigr)$. It tells the online corrector which past prediction a new probe reading verifies. Finally, it makes the cumulative oxygen a natural coordinate for comparing heats. Two heats that have consumed the same oxygen have reached the same stage of the blow, even if their flows differ (Figure 6).

> **[Figure 6 — PLACEHOLDER]** 时间映射示意图：上方时间轴标出 bath event at $t$（火焰立即响应）与 analyzer reading at $t_{\mathrm{arr}}$；下方为累积耗氧坐标，标出 $V(t)$ 与 $V(t)+\Delta V_{\mathrm{pipe}}$；用两种流量（低/高）展示同一 $\Delta V_{\mathrm{pipe}}$ 对应不同时延。宽度 0.8。
> Caption: **Arrival-time map.** The duct must be refilled by $\Delta V_{\mathrm{pipe}}$ before the gas generated at $t$ reaches the analyzer, so a larger lance flow gives a shorter delay.

---

## 5 Task 1: Spectral Outlier Removal and Continuous Reconstruction

Task 1 asks for two things: the removal of outliers and a continuous surface that can be drawn in three dimensions. The data make both harder than they appear. A heat contains about 2.9 million intensity values, the flame flickers strongly from frame to frame, a few pixels saturate, and two narrow emission lines vary on their own. Filters that work along time only treat the flicker as outliers and ignore the strong correlation between neighbouring wavelengths. Ordinary principal component analysis uses this correlation but is distorted by the outliers it should detect. We therefore separate a low-rank signal from sparse outliers explicitly and build the continuous surface from the low-rank part.

### 5.1 Rank-Constrained Robust Low-Rank Model (RLR)

A flame spectrum is a smooth thermal continuum with a few emission lines. As the blow proceeds, the temperature and the line intensities change, but the spectral shapes stay within a small family. The log-intensity matrix of a heat is therefore nearly low-rank, while spikes and corrupted frames are sparse. RLR uses this structure in four steps.

**Step 1: Log transform and censoring.** We set $X_{t\lambda} = \ln I(t,\lambda)$ on the unified grid and mark saturated pixels as missing. The log transform turns multiplicative flicker into an additive shift, which a single component can absorb.

**Step 2: Low-rank plus sparse decomposition.** We decompose

$$
\mathbf{X} = \mathbf{L} + \mathbf{S} + \mathbf{N}, \qquad \operatorname{rank}(\mathbf{L}) \le r, \tag{3}
$$

where $\mathbf{S}$ holds the outliers and $\mathbf{N}$ is small noise. Classical principal component pursuit [1] does not separate the noise here: its low-rank part reaches a rank of 426–446 in every heat. We therefore fix the rank and alternate two projections [2]. Given $\mathbf{S}$, $\mathbf{L}$ is the best rank-$r$ approximation of $\mathbf{X}-\mathbf{S}$; given $\mathbf{L}$, $\mathbf{S}$ keeps the residuals that exceed five times a robust scale. The iteration stops when the set of outliers no longer changes.

**Step 3: Outlier rules.** The robust scale $\sigma_\lambda$ is estimated separately for each wavelength, because the ultraviolet and near-infrared ends carry fewer counts and more noise. A pixel is an outlier if $|X_{t\lambda}-L_{t\lambda}| > 5\sigma_\lambda$. A frame is anomalous if its mean standardized residual has a robust z-score above 5. Within such a frame, we remove only the pixels with $|z|>2.5$ and keep the rest, so that the frame still determines its own low-rank scores. A run longer than 5 s is treated as a change of operation, such as the end of the blow, rather than as an outlier.

**Step 4: Continuous surface.** After removing outliers, we complete the matrix with rank $r$ and write

$$
\hat{I}(t,\lambda) = \exp\Bigl[\mu(\lambda) + \sum_{j=1}^{r} a_j(t)\,\varphi_j(\lambda)\Bigr], \tag{4}
$$

where $\mu$ is the mean log spectrum, the loadings $\varphi_j$ are interpolated by cubic splines and the scores $a_j$ by smoothing splines. Equation (4) can be evaluated at any time and wavelength, and it stays positive by construction. The smoothing parameter of $a_j$ is selected by generalized cross-validation for the reconstruction. The three-dimensional figures use a stronger smoothing ($\lambda_s = 10^4$) to display the trend of the blow.

### 5.2 Model Solution and Results

#### 5.2.1 Low-Rank Structure and Outlier Detection

The first three components explain at least 99.6% of the log-spectral variance in every heat, with a mean of 99.8%. We choose the rank by hiding 10% of the valid pixels. The hold-out error falls from 0.099 at $r=1$ to 0.032 at $r=3$ and rises again for larger ranks, because the additional components begin to fit noise. We therefore use $r=3$. RLR flags 0.65–3.6% of the pixels and 33–117 isolated frames per heat.

The components have a clear physical meaning. In every heat, the score of the first component follows the logarithm of the total intensity with $|r| = 1.00$, so this component measures the overall brightness of the flame. By contrast, the second score is almost independent of brightness ($|r| = 0.03$). Instead, it correlates with the intensity ratio between 800 and 500 nm ($|r| = 0.52$ on average) and with the contrast of the potassium line ($|r| = 0.48$). This component therefore describes the colour of the flame and the strength of its emission lines, two quantities related to the flame temperature.

Figure 7 shows the decomposition for Heat 3. The low-rank part keeps the rise of the flame during ignition, the plateau of the main blow and the decay near the end. The sparse part concentrates on two narrow bands at 767 nm and 589 nm, the two strongest emission peaks of the mean spectrum. These lines fluctuate independently of the continuum and may correspond to the potassium doublet (766.5/769.9 nm) and the sodium D lines.

> **[Figure 7]** `B_results/figures/q1_rpca_decomposition.pdf`，宽度 1.0。
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

> **[Figure 8]** `B_results/figures/q1_outlier_example.pdf`，宽度 1.0。
> Caption: **Outlier detection and reconstruction.** Left: intensity at 600 nm with detected outliers and the continuous reconstruction. Right: an anomalous frame, rebuilt from its retained pixels.

#### 5.2.3 Continuous 3D Spectral Surfaces

Figure 9 presents the continuous spectral surfaces of the ten heats. All heats share the same pattern. A dark ignition stage of 1–2 min ends with a sharp rise as the main blow starts. A broad plateau with its maximum between 750 and 900 nm follows, and the intensity declines at the end of the blow. The 6C heats reach higher intensities than the 6B heats and show more saturation, which agrees with the different instrument settings identified in Section 3.1. The timing of the rise and the decline matches the stage breakpoints found from the gas data in Section 6.2, which shows that the flame and the gas describe the same process.

> **[Figure 9]** `B_results/figures/q1_surface_grid.pdf`，宽度 1.0，占半页以上。单炉大图 `q1_surface_heat01–10.pdf` 放附录。
> Caption: **Continuous spectral surfaces of the ten heats (trend surfaces, log₁₀ intensity).** Every heat shows ignition, a main-blow plateau and an end-of-blow decline; the 6C heats are brighter.

> 数字出处（不进论文）：A 版 `q1_outlier_summary.csv`、`q1_rank_selection.csv`、`q1_injection_test.csv`、`q1_reconstruction_error_summary.csv`、`q1_frame_holdout_summary.csv`、`q1_emission_peaks.csv`；成分物理含义为本轮由 `data_clean/heat*_spec.npz` 计算（第 1 成分得分与 log 总光强 |r| = 1.00；第 2 成分与亮度 |r| = 0.03、与 800/500 nm 比 0.52、与钾线对比度 0.48，均为十炉平均）。

---

## 6 Task 2: Oxygen–Gas Relationship and Variable Flue-Gas Delay

Task 2 contains three questions of increasing depth: how to recover continuous process curves, how the gas depends on the oxygen consumption, and how the delay depends on the lance flow. The first two describe the gas as a function of the blow; the third needs a model of the duct. We answer them in this order, because the stage structure found in the second step also locates the events used to check the delay.

### 6.1 Continuous Reconstruction of the Process Signals

The lance flow $Q(t)$ comes from the 24-s regression slope of Section 3.3, and $V(t)$ is the monotone cubic interpolation of the counter updates. For the gas readings, we remove two kinds of outliers. Stuck readings are runs that stay constant for more than 10 s during the active blow; we find five such runs of 22–94 points. Spikes are detected by a Hampel filter applied to the sequence of value updates, with a floor equal to the measurement noise, which flags at most four points per heat. The cleaned readings are reconstructed by monotone cubic interpolation followed by a smoothing spline. Hiding 10% of the updates gives a reconstruction error of 0.20 for CO (mean 31.0) and 0.10 for CO₂ (mean 13.8), both below 1%. The cumulative emissions are the trapezoidal integrals of the reconstructed readings.

Figure 10 shows the six curves of Heat 3. The flow holds at about 2.4 m³/s during ignition, steps up to about 4.8 m³/s in the main blow and rises again near the end. CO₂ peaks within the first three minutes, whereas CO rises later and stays high until the end of the blow. The cumulative emissions therefore grow in opposite orders: the CO₂ curve rises early and flattens, and the CO curve starts slowly and steepens.

> **[Figure 10]** `B_results/figures/q2_curves_heat03.pdf`，宽度 1.0。其余九炉 `q2_curves_heat*.pdf` 放附录。
> Caption: **Continuous process curves of Heat 3:** lance flow, cumulative oxygen, CO, CO₂ and their cumulative emissions. CO₂ peaks early and CO late; dotted lines mark the stage breakpoints.

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

The four stages follow the chemistry of the blow. In Stage I, silicon and manganese consume much of the oxygen, and the small amount of carbon that burns at the low bath temperature leaves mostly as CO₂. In Stage II, the bath heats up and decarburization accelerates, so the CO reading rises by 0.055 percentage points per cubic metre of oxygen. In Stage III, the CO reading stays nearly constant while about three quarters of the carbon gas is CO. In Stage IV, carbon is depleted and CO falls. The cumulative emissions follow the same breakpoints: a piecewise-linear fit of $E_{\mathrm{CO}}(V)$ and $E_{\mathrm{CO_2}}(V)$ reaches $R^2 = 0.998$ and a lower AIC than logistic and quadratic fits. Between the two converter groups, only the slope of Stage III differs significantly (Mann–Whitney test, $p = 0.02$).

> **[Figure 11]** `B_results/figures/q2_relation_instant.pdf`，宽度 1.0。
> Caption: **CO and CO₂ versus cumulative oxygen for the ten heats.** Three breakpoints divide every blow into the same four stages; dashed lines are the piecewise-linear fits.

### 6.3 Plug-Flow Delay Model (PFD)

To estimate $\Delta V_{\mathrm{pipe}}$ in Equation (1), we need a signal whose shape is common to all heats, so that the shift between heats and their own oxygen clock can be identified. We use the oxygen-equivalent gas signal $y_{\mathrm{ox}} = y_{\mathrm{CO_2}} + 0.5\,y_{\mathrm{CO}}$, which measures the oxygen bound in the carbon gas. By the balance of Section 3.4, its time integral is proportional to the oxygen supplied, with a ratio that varies by only 6.5% between heats. In the oxygen coordinate, the ten heats indeed share one trajectory (Figure 12, middle). PFD describes this trajectory in three steps.

**Step 1: Shifted generation curve.** The gas reaching the analyzer at cumulative oxygen $V$ was generated at $V^{+} = \max(V-\Delta V_{\mathrm{pipe}},\,0)$. We write the generation curve as

$$
u(V) = g(V)\,\bigl[\eta + (1-\eta)\,s\bigl((V^{+}-V_{\mathrm{m}})/w_{\mathrm{m}}\bigr)\bigr]\; s\bigl(-(V^{+}-V_{\mathrm{e}})/w_{\mathrm{e}}\bigr), \tag{5}
$$

where $s(\cdot)$ is the logistic function and $g(V) = s\bigl((V-\Delta V_{\mathrm{pipe}})/3.5\bigr)$ switches the signal on when the first gas arrives. The parameter $\eta$ is the ignition level, and the pairs $(V_{\mathrm{m}}, w_{\mathrm{m}})$ and $(V_{\mathrm{e}}, w_{\mathrm{e}})$ locate the main-blow rise and the end-of-blow decline.

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

> **[Figure 12]** `solution_v4_verified/solution_outputs/figures/q2_variable_delay_and_diagnostics.pdf`，宽度 1.0。
> Caption: **Plug-flow delay model.** Left: $\tau(Q)$ with the 95% bootstrap band and the ignition and main-blow onset events. Middle: oxygen-equivalent readings of the ten heats collapse onto one trajectory in the oxygen coordinate. Right: apparent scale factor of each heat.

#### 6.4.2 Validation against Main-Blow Onset Events

The value of $\Delta V_{\mathrm{pipe}}$ is anchored mainly by the arrival of the first gas after ignition. Because ignition itself takes some time, we check the model against the main blow (Table 7). When the lance flow first exceeds 3.5 m³/s, the oxygen-equivalent signal crosses 13 vol.% after 28–45.5 s in the ten heats, with a median of 34.5 s. These event times include the build-up of decarburization, so they bound the transport delay from above. For the mean flows of 4.4–6.6 m³/s over these intervals, Equation (6) gives 24–34 s, below the bound as expected. A regression of all twenty ignition and main-blow events on $1/Q$ reaches $R^2 = 0.74$.

#### 6.4.3 Model-Free Check by Windowed Cross-Correlation

This check uses no transport model at all. A cross-correlation between the first spectral component and CO in 120-s windows gives 44 windows with a clear peak. Their delay decreases with the window-mean flow (Spearman $\rho = -0.75$, $p = 3.4\times10^{-9}$), from 63 s at 4.5–5.0 m³/s to 14.5 s above 5.5 m³/s (Figure 13). Because the window time correlates only weakly with the flow ($\rho = -0.12$), the trend does not come from the stage of the blow.

> **Table 7. Delay estimates from independent methods**

| Method | Flow range (m³/s) | Delay (s) |
|---|---|---|
| PFD, Equation (6) | 5.0 | 30.4 (29.7–31.5) |
| Main-blow onset events (10 events, upper bound) | 4.4–6.6 | 28–45.5, median 34.5 |
| Windowed cross-correlation (44 windows) | 3.2–5.7 | median 40.5; decreases with $Q$, $\rho=-0.75$ |
| Constant delay maximizing spectral predictability | full blow | 31 |
| First gas after ignition (upper bound) | 2.35 | median 49.8 |

> **[Figure 13 — 需修改后使用]** `B_results/figures/q2_delay_vs_flow.pdf`，宽度 0.7。**排版前须重画**：去掉图中 A 版的 170/Q 曲线与其置信带，改画 PFD 曲线 $\tau = 5 + 127.1/Q$，保留窗口散点与分档中位数，避免与正文的时延模型矛盾。
> Caption: **Windowed delay estimates versus lance flow.** Dots are 44 window estimates and squares are the medians of four flow bins with interquartile ranges; the curve is the PFD model of Equation (6).

The windowed estimates fall faster with the flow than $1/Q$ does. This suggests that the duct volume is not the only factor; a higher flow may also enhance mixing and post-combustion in the hood. In the main-blow flow range of 4.4–6.6 m³/s, Equation (6) gives 24–34 s. These values stay below the two event-based upper bounds and agree with the constant delay of 31 s that maximizes spectral predictability. We therefore adopt Equation (6) as the delay model for Task 3 and test the consequence of a ±10% error in $\Delta V_{\mathrm{pipe}}$ in Section 8.

> 数字出处（不进论文）：B 版 `q2_delay_vs_flowrate_table.csv`、`q2_tmix_identifiability_sensitivity.csv`、`q3_per_fold_delay_params.csv`、`q2_step_response_delay_events.csv`、`q2_mass_balance_consistency.csv`、`q2_detrended_residual_corr.csv`、`verification_audit_report.csv`（合成回收 135.29 m³）；A 版 `q2_stages.csv`、`q2_stage_group_test.csv`、`q2_cumulative_fit.csv`、`q2_gas_holdout_error.csv`、`q2_reconstruction_summary.csv`、`q2_delay_auxiliary.csv`、`q2_delay_window_bins.csv`、`q2_delay_model_comparison.csv`。

---

## 7 Task 3: Real-Time Off-Gas Prediction from Flame Spectra

Task 3 combines the spectral basis of Section 5 with the delay model of Section 6. It differs from an ordinary regression in four ways. The target is delayed, so the flame at time $t$ must be matched with the gas that the analyzer reports $\tau$ seconds later. Only ten heats are available, and a model learned on one heat fails on another. The problem requires the heat-wise models to be combined. Finally, a real-time model may use only past data, and its evaluation must not let the test heat leak into any tuning step. Our soft sensor addresses these four points in turn.

### 7.1 Prediction Target and Causal Features

At time $t$, the flame reflects the bath state at $t$, but the analyzer will report this state only at $t_{\mathrm{arr}}(t)$. The target of real-time prediction is therefore the delay-aligned reading $y\bigl(t_{\mathrm{arr}}(t)\bigr)$ from Equation (1). The prediction is available about 25–30 s before the analyzer reports the same gas.

All features use data up to time $t$ only. For each frame, we subtract the dark level below 385 nm, smooth along wavelength and resample onto 127 points between 380 and 1010 nm. We then project the frame onto the eight-dimensional spectral basis learned in Task 1 from the training heats; saturated pixels are excluded from the projection. The projection replaces 127 correlated intensities by eight scores and transfers the cleaning of Task 1 to new frames.

Three physical descriptors complement the scores. A colour temperature follows from the Wien approximation: on three continuum windows that avoid the emission lines (480–575, 610–750 and 785–960 nm), the slope of $\ln(I\lambda^{5})$ against $1/\lambda$ equals $-c_2/T_{\mathrm{c}}$ with $c_2 = 14{,}388$ µm·K. The spectral centroid and the spectral entropy describe where and how widely the radiation is distributed. Exponential moving averages with time scales of 5, 25, 80 and 200 s summarize the recent history, from flicker-free current values to the trend of the stage. The process features include $V(t)$, the causal lance flow and the blowing progress.

### 7.2 Dual-Modal Soft Sensor (DSS)

**Step 1: Linear-tree predictor.** A ridge regression on bounded linear features captures the trend and extrapolates beyond the training range. This matters for Heat 1, whose CO peak of 59 exceeds that of all other heats. Gradient-boosted trees [7] then model the residual nonlinearity.

**Step 2: Model combination.** Models trained on single heats do not transfer: when a model trained on one heat predicts another, the median R² is −0.02, and 52% of the pairs are negative. We therefore combine the heats rather than select one. Three schemes follow the combination requirement. The pooled model uses all training heats, the grouped models use one converter group or one cluster of heats, and the soft-gated model weights each training heat by its similarity to the test heat. For the soft-gated model, let $\mathbf{d}_j$ be a fingerprint of heat $j$ computed from its first 60 s: the mean normalized spectrum, the mean Task-1 scores, the mean flow and the mean log total intensity. The sample weight of training heat $j$ is

$$
w_j = 0.55 + 0.45\,M\,\pi_j, \qquad \pi_j \propto \exp\Bigl(-0.65\,\frac{\lVert\mathbf{d}_j-\mathbf{d}_{\mathrm{test}}\rVert^2}{\operatorname{med}_i\lVert\mathbf{d}_i-\mathbf{d}_{\mathrm{test}}\rVert^2} + 0.55\,\mathbb{1}_{\text{same group}}\Bigr), \tag{7}
$$

where $M$ is the number of training heats and $\sum_j \pi_j = 1$. The uniform part keeps every heat in the training set, and the similarity part gives more weight to heats that started the blow in a similar way. The fingerprint uses only the first 60 s of the test heat, so the weights are available in real time.

**Step 3: Validation protocol.** The outer loop is leave-one-heat-out (LOHO): each of the ten heats is predicted by a model that never saw it. Every quantity learned from data, including the spectral basis and $\Delta V_{\mathrm{pipe}}$, is recalibrated within each fold. An inner eight-versus-one loop selects the correction gain and the interval width, so the test heat never enters a tuning step. Two automated checks confirm this design: corrupting the future frames of a heat leaves its past predictions unchanged, and corrupting the test heat leaves every trained quantity unchanged.

**Step 4: Smith-type correction.** A probe reading taken at time $t$ verifies the prediction made at $t_{\mathrm{arr}}^{-1}(t)$, in the spirit of the Smith predictor for dead-time processes [8]. The difference between the reading and that earlier prediction updates a bias term, which corrects the current prediction:

$$
b_k = \alpha\,e_k + (1-\alpha)\,b_{k-1}, \qquad \hat{y}_{\mathrm{c}}(t) = \hat{y}(t) + g\,b_k, \tag{8}
$$

where $e_k$ is the latest verified error, $\alpha$ is a smoothing factor and $g$ is the gain selected in the inner loop. With a probe interval $N_{\mathrm{p}}$, the bias is updated only when a reading arrives.

**Step 5: Prediction interval.** Split conformal prediction [9] scales the inner-loop residuals by a local spread to give a 90% prediction interval.

### 7.3 Model Solution and Results

#### 7.3.1 Ablation and Model Combination

Table 8 compares the variants under LOHO, where higher R² and lower RMSE are better. Delay alignment is essential: without it, the spectra and process features predict the delay-aligned CO₂ with a mean R² of only 0.33; with it, the R² rises to 0.66. The pooled model and the soft-gated model perform almost equally (CO: 0.881 and 0.873), while hard grouping loses accuracy because each group leaves only three to five training heats. We adopt the soft-gated model as the zero-probe DSS, because it satisfies the combination requirement and serves as the base of the Smith correction.

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

Figure 14 shows the LOHO predictions for the ten heats. The zero-probe DSS follows the ignition, the ramp-up and the end-of-blow decline in every heat. Its CO R² exceeds 0.85 in seven heats and ranges from 0.67 to 0.97 overall. The CO₂ predictions are weaker in Heats 5 (0.33), 1 (0.39) and 9 (0.50). Heat 1 is the only heat whose CO peak lies above the training range, and Heat 5 combines the coarser 1-s sampling with the strongest saturation.

The conformal intervals behave as intended. Without probe readings, the 90% intervals have a mean half-width of 10.1 for CO and contain 87.0% of the readings. The coverage falls to 62% in Heats 1 and 3 and reaches 76–100% in the other heats. With continuous correction, the half-width shrinks to 5.7, and the intervals contain 89.8% of the CO readings and 89.75% of the CO₂ readings, close to the nominal level.

> **[Figure 14]** `solution_v4_verified/solution_outputs/figures/q3_all10_loho_predictions.pdf`，宽度 1.0，占大半页。
> Caption: **Leave-one-heat-out predictions of CO and CO₂ for the ten heats** with 90% conformal intervals. The prediction follows every stage; Heat 1, whose CO peak exceeds the training range, is the hardest case.

#### 7.3.3 Reducing Probe Usage

The DSS changes the role of the probe from a continuous sensor to an occasional calibrator. Figure 15 plots accuracy against the probe interval. With one reading every 120 s, the corrected prediction reaches CO R² = 0.908 and CO₂ R² = 0.695. The lagged reading held over the same interval gives only 0.593 and −1.04. Even with continuous readings, the corrected prediction beats the lagged reading (0.949 versus 0.916 for CO), because it removes the 30-s delay.

The trade-off offers a practical choice. A 60-s interval saves 99.2% of the readings and keeps CO R² at 0.920; a 30-s interval saves 98.3% and reaches 0.934. In a partial least squares model, the most important wavelengths are 765–770 nm and 590 nm, which are the potassium and sodium lines found in Task 1. The red and near-infrared continuum at 730–825 nm ranks next. The same emission lines thus separate outliers in Task 1 and carry information on the off-gas in Task 3.

> **[Figure 15]** `solution_v4_verified/solution_outputs/figures/q3_transfer_pareto_and_vip.pdf`，宽度 1.0。
> Caption: **Probe saving versus accuracy and wavelength importance.** Left: with one reading every 120 s, the corrected prediction keeps CO R² above 0.9, while the lagged reading drops to 0.59. Right: the potassium and sodium lines carry the most information.

> 数字出处（不进论文）：B 版 `q3_loho_ablation_summary.csv`、`q3_loho_per_heat_metrics.csv`（Combined_SoftGated_Hybrid 逐炉 CO：0.673/0.775/0.806/0.939/0.897/0.959/0.856/0.962/0.970/0.891；CO₂ 最弱为 H05 0.333、H01 0.386、H09 0.495）、`q3_single_heat_transfer_matrix.csv`、`q3_conformal_90pct_coverage.csv`（零探头 CO 逐炉 0.62–1.00、均值 0.870、半宽 10.14；连续校正半宽 5.65）、`q3_transition_regime_rmse.csv`、`q3_pls_vip_wavelength.csv`、`verification_audit_report.csv`（第 5、6 项泄漏检验）；颜色温度与指纹定义见 `q3_fusion_predict.py`。

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

The most important check concerns Task 3, because the delay defines its target. We let the system use a duct volume 10% too small or too large, for training and for the Smith correction, while the evaluation still uses the calibrated delay. The zero-probe accuracy changes by at most 0.003, and the corrected CO accuracy by at most 0.003. The CO₂ correction improves when the assumed volume grows. This agrees with the spectral-predictability criterion, whose optimal constant lag is 72 s for CO₂ but 31 s for CO, and suggests that CO₂ mixes over a longer path than CO.

> 数字出处（不进论文）：A 版 `q1_threshold_sensitivity.csv`、`q1_rank_selection.csv`、`q2_delay_model_comparison.csv`（CO₂ 最优常数时延 72 s）、`q3_delay_misestimation_sensitivity.csv`（本次补算，脚本 `B_results/code/sens_delay_misestimation.py`）；B 版 `q2_tmix_identifiability_sensitivity.csv`、`q2_delay_vs_flowrate_table.csv`、`q3_loho_ablation_summary.csv`。

---

## 9 Model Evaluation and Further Discussion

### 9.1 Strengths

- **Data-consistency diagnosis.** The analysis identifies the corrected Attachment 1, the 1-s step of Heat 5, the beat of the oxygen counter and the volume-fraction nature of the gas readings before any model is fitted.
- **One low-rank structure for cleaning and continuity.** RLR uses the same three components to detect outliers, fill censored pixels and build the continuous surface, and the injection test verifies each function.
- **A delay model with a physical basis.** PFD derives the flow dependence from the plug-flow assumption. Its main-blow delay agrees with an independent constant-delay estimate and stays below the event-based upper bounds, and the windowed analysis confirms that the delay decreases with the flow.
- **Strictly causal and leakage-free validation.** Every learned quantity is recalibrated within each LOHO fold, and the test heat never enters a tuning step.
- **A practical soft sensor.** The DSS predicts the off-gas about 30 s ahead and keeps CO R² above 0.9 with 99.6% fewer probe readings.

### 9.2 Weaknesses

- **Ten heats.** Heat 1 lies outside the range of the other heats, and its interval coverage falls to 62% without probe readings. More heats, especially high-carbon ones, would widen the training range of the soft sensor.
- **Narrow flow range in the main blow.** The median lance flow of every heat lies between 4.6 and 5.1 m³/s, so the flow dependence of the delay rests on a small number of flow changes. Under Assumption 1, the mixing constant cannot be identified separately; deliberate flow steps would resolve it.
- **Unit of the gas readings.** Assumption 4 rests on the balance check rather than on instrument documentation, so the absolute flue-gas volume remains unknown. A calibration record of the analyzer would remove this uncertainty.

### 9.3 Further Discussion

Additional heats with deliberate lance-flow steps would identify $T_{\mathrm{mix}}$ and test a power-law delay, which the windowed estimates suggest. A radiometric calibration of the spectrometer would turn the colour temperature into an absolute flame temperature. The soft sensor can also be extended to the endpoint: combining the predicted CO with the decline of Stage IV would give an early estimate of the end of decarburization.

The framework transfers to other converters with little change. Equation (1) needs only the cumulative oxygen and one calibrated volume, and the spectral basis can be relearned from a few heats. The similarity weights of Equation (7) then let a new heat borrow from the most similar past heats. In operation, the probe interval can be chosen from Figure 15 to balance probe wear against the required accuracy, and the conformal interval signals when a heat moves outside the experience of the model.

---

## 10 Conclusions

We built a delay-aware framework that fuses the real-time flame spectrum with the delayed flue-gas signal of a converter. A diagnosis of the data corrected Attachment 1, fixed the time step of Heat 5, removed the beat of the oxygen counter and showed that the gas readings are volume fractions.

For Task 1, the Rank-Constrained Robust Low-Rank model separated the spectra into three physical components and sparse outliers. It detected injected spikes with an F1 score of 0.89, reconstructed held-out pixels with a relative error of 2.0%, and produced continuous spectral surfaces for all ten heats.

For Task 2, the reconstructed process curves revealed four blowing stages with breakpoints at 23%, 43% and 90% of the total oxygen. The Plug-Flow Delay model gave $\tau(Q) = 5 + 127.1/Q$ s, from 47.4 s at 3 m³/s to 30.4 s at 5 m³/s, and independent estimates confirmed that the delay decreases with the flow.

For Task 3, the Dual-Modal Soft Sensor combined models of different heats through similarity weighting. Without probe readings, it predicted the delay-aligned CO and CO₂ with mean R² of 0.873 and 0.658 on all ten heats; with one probe reading every 120 s, it reached 0.908 and 0.695. The flame spectrum can therefore stand in for most probe readings and deliver the off-gas information about 30 s earlier.

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

（待补：转炉炉口火焰光谱与终点控制、转炉烟气分析时延方面的 2–3 篇领域文献，需逐条核实后加入。）

---

## Appendix（不计入正文页数）

- A. Software: Python 3.13; NumPy, SciPy, pandas, scikit-learn, XGBoost, LightGBM, Matplotlib.
- B. Source code: `B_results/code/`（Task 1 与数据诊断、Task 2 分阶段与窗口时延、敏感性补算）与 `solution_v4_verified/solution_code/`（Task 2 PFD 与 Task 3 DSS）。
- C. Additional figures: single-heat spectral surfaces (`q1_surface_heat01–10`), single-heat process curves (`q2_curves_heat01–10`).
- D. Report on use of AI（若官方要求）。
