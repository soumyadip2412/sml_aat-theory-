# Regression Assumption Testing and Model Diagnostics

Notebook: `notebooks/AirWatch_Bengaluru_SML_AAT.ipynb` §6-7 (HC3 standard errors, the ACF plot, the top-days table and the sensitivity refit are in the archived `archive/original_notebooks/04_diagnostics.ipynb`) · Figures 03-05 · Tables `assumption_tests.csv`, `vif.csv`, `robust_se_comparison.csv`, `influence_*.csv`, `top_influential_days.csv`

The model checked is **MLR 2** (8 pollutants + season), fitted on the 1181 training days.

## 6. Regression Assumption Testing

### Formal tests: before and after the log transform
For each test, H₀ is that the assumption holds, so p < 0.05 means a violation. Durbin-Watson ≈ 2 means no autocorrelation.

| Assumption | Test | MLR 2 (PM2.5) | MLR 2 (log PM2.5) |
|---|---|---|---|
| Linearity | Rainbow, p | 0.0003 ✘ | 0.105 ✔ |
| Normality | Residual skew | 2.59 | −0.20 |
| Normality | Jarque-Bera, p | < 0.001 ✘ | < 0.001 ✘ |
| Normality | Shapiro-Wilk, p | < 0.001 ✘ | < 0.001 ✘ |
| Homoscedasticity | Breusch-Pagan, p | < 0.001 ✘ | < 0.001 ✘ |
| Homoscedasticity | White, p | < 0.001 ✘ | < 0.001 ✘ |
| Independence | Durbin-Watson | 0.999 ✘ | 0.904 ✘ |
| Independence | Ljung-Box (lag 7), p | < 0.001 ✘ | < 0.001 ✘ |
| Fit | Adjusted R² | 0.563 | **0.609** |

### Graphical checks (Figure 09)
- **Residuals vs fitted:** the cloud is slightly curved, and its spread widens as the fitted value grows, a mild funnel shape.
- **Normal Q-Q plot:** the right tail bends upward, because a few spike days are much higher than the model predicts.
- **Residuals over time:** residuals drift in runs of positive and negative values instead of scattering randomly. This is autocorrelation.

### Multicollinearity: VIF
| Predictor | NO | NO2 | NOx | PM10 | O3 | SO2 | NH3 | CO |
|---|---|---|---|---|---|---|---|---|
| VIF | 2.22 | 1.79 | 1.71 | 1.41 | 1.37 | 1.37 | 1.12 | 1.03 |

All VIFs are below 2.5, well under the usual cut-off of 5. **There is no serious multicollinearity**, so the standard errors are not inflated and all 8 predictors are kept. The negative NO coefficient comes from the chemical overlap NOx ≈ NO + NO2, not from instability.

### Remedy 1: log transform (non-linearity, skewness, impossible predictions)
Modelling log(PM2.5) treats pollutant effects as **multiplicative** (percentage changes), which suits concentration data. The results:
- The Rainbow test **no longer rejects linearity** (p = 0.105).
- Residual skew falls from 2.59 to −0.20, and the histogram becomes symmetric (Figure 10).
- Adjusted R² rises from 0.563 to 0.609.
- Every back-transformed prediction is positive.

The formal normality tests still reject. With n = 1181 these tests detect even tiny departures; the Q-Q plot shows the remaining issue is only a handful of extreme days. By the central limit theorem, OLS inference stays approximately valid at this sample size.

**Log-model coefficients as % change in PM2.5** (exp(β) − 1):
- **Monsoon vs winter:** −30%.
- **Per 10 µg/m³ rise, holding the others constant:** NOx +8.3%, NO2 +4.9%, NH3 +4.4%, PM10 +4.3%, O3 +4.0%.

### Remedy 2: heteroscedasticity, with White (HC3) robust standard errors
The Breusch-Pagan and White tests still reject constant variance after the log transform. Unequal variance does not bias the coefficients, but it makes the ordinary standard errors wrong. HC3 standard errors (`robust_se_comparison.csv`) are valid under heteroscedasticity; they are up to about 25% larger than the OLS ones.

### Remedy 3: autocorrelation, with the Durbin-Watson test, Cochrane-Orcutt and Newey-West
- **Durbin-Watson:** d = 0.904. This is far below the 5% lower bound d_L ≈ 1.86 (n ≈ 1180, k = 11), so there is **positive autocorrelation**. The estimated ρ̂ = 1 − d/2 ≈ 0.55: today's error carries about half of yesterday's. The residual ACF (Figure 11) confirms it: correlations stay significant up to about lag 5-6 days, with a smaller bump around 14-15 days.
- **Cochrane-Orcutt / feasible GLS (GLSAR, AR(1)):** estimates ρ = 0.579. On the transformed model, **Durbin-Watson = 2.07**, so the autocorrelation is removed.
- **Newey-West (HAC) standard errors:** these are 1.3 to 2.4 times the OLS standard errors. OLS was therefore over-confident.

**Which predictors survive every correction?** They are significant under OLS, HAC and GLSAR alike: **PM10, NOx, NO, NH3, SO2, O3 and Monsoon**. NO2 is borderline under GLSAR (p = 0.050). **CO is not reliable:** it is significant under OLS (p = 0.007) but not under HAC (0.079) or GLSAR (0.65).

## Model Diagnostics: outliers and influential observations

All diagnostics use the log model (n = 1181, p = 12 parameters).

| Measure | Detects | Threshold | Days flagged |
|---|---|---|---|
| Studentized deleted residual | Outlying Y | \|tᵢ\| > 4.11 (Bonferroni, α = 0.05) | 3 (0.3%) |
| Leverage hᵢᵢ (hat matrix) | Outlying X | hᵢᵢ > 2p/n = 0.020 | 79 (6.7%) |
| DFFITS | Influence on own fitted value | \|DFFITS\| > 2√(p/n) = 0.20 | 74 (6.3%) |
| Cook's distance | Influence on all fitted values | D > 4/n = 0.0034 | 72 (6.1%) |
| Cook's distance (strong) | Highly influential | D > F₀.₅(p, n−p) = 0.95 | **0** |
| DFBETAS | Influence on any one coefficient | \|DFBETAS\| > 2/√n = 0.058 | 268 (22.7%) |

### Findings
- **Added-variable plots (figure `04_added_variable_plots.png`, full log model including season):** PM10, NOx, NO2 and O3 keep a clear linear relationship with log(PM2.5) after adjusting for the others, and NO a clear negative one. NH3 and SO2 have weak slopes that lean on a few high-X days (the outlying-X points flagged by leverage). The CO panel is almost flat, which agrees with CO failing the robust tests. None of the panels shows curvature.
- **No single day dominates the fit.** The largest Cook's distance is 0.056, far below the "strong" cut-off of 0.95.
- **The most influential days** (Table `top_influential_days.csv`) are unusually *clean* winter days, such as 28-29 Jan 2017 (PM2.5 of 10 and 7), where the model expected winter-level pollution. The 13-Dec-2016 spike (PM2.5 = 204) is also influential.
- **Diagnostics caught sensor errors.** In the first run of this analysis, the most influential days were 3-8 July 2017. The cause was one impossible reading (PM2.5 = 314 > PM10 = 120) that interpolation had spread into a 5-day artificial spike. Checks for PM2.5 > PM10 (64 days) and implausible CO (7 days) were then added to the data cleaning. All results in this report use the corrected data.
- **Sensitivity check:** removing the 72 Cook's-flagged days raises adjusted R² from 0.609 to 0.706. Every pollutant coefficient keeps its sign and significance; the largest change is NO2 (+46%). These days are real pollution events, not errors, so **they are kept in the model**. The conclusions do not depend on them.

## Out-of-sample effect of the remedy
The log model's predictions were back-transformed with Duan's smearing factor (1.050) to correct for re-transformation bias.

| Model | RMSE (test) | MAE (test) | R² (test) | Lowest prediction |
|---|---|---|---|---|
| MLR 2 (PM2.5) | 8.57 | 6.86 | 0.584 | 10.7 |
| MLR 2 (log PM2.5) | **8.05** | **6.48** | **0.633** | 16.1 |

The log transform improves both the assumptions and the test-period accuracy.

## Summary for the report
1. Linearity: violated by the linear model, **fixed** by the log transform.
2. Normality: much improved (skew 2.6 → −0.2); the remaining tail is a few extreme days, and inference is acceptable at n = 1181.
3. Homoscedasticity: violated; handled with **HC3 robust standard errors**.
4. Independence: strongly violated (DW = 0.90, ρ ≈ 0.55); handled with **Cochrane-Orcutt GLS** (DW → 2.07) and **Newey-West standard errors**.
5. Multicollinearity: **not a problem** (all VIF < 2.5).
6. Influence: no highly influential points; the conclusions are robust. The diagnostics also exposed sensor errors that were then fixed in the data cleaning.
