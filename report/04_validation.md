# Model Selection and Estimator Validation

Notebook: `notebooks/AirWatch_Bengaluru_SML_AAT.ipynb` §8 (AIC, BIC, Cp). Cross-validation and bootstrap (Sections 7b-7c) are beyond the syllabus and are in the archived `archive/original_notebooks/05_validation_model_selection.ipynb` · Figures 16-18 (archived) · Tables `all_subsets.csv`, `best_subset_*.csv`, `cross_validation_*.csv`, `coefficient_stability_folds.csv`, `bootstrap_*.csv`

All of this work uses the 1181 training days only; the test period stays untouched. The target is log(PM2.5), the remedied model from Section 6.

## 7a. Model selection: AIC, BIC and Mallow's Cp

**Method:** all-subsets regression was run over every combination of the 8 pollutants, with and without Season (Season counts as one block of 3 dummies). That gives 2⁹ − 1 = **511 candidate models**. For each model with p parameters:
- AIC = n·ln(SSE/n) + 2p
- BIC = n·ln(SSE/n) + p·ln(n)
- Cp = SSE_p / MSE_full − (n − 2p)

### Best model of each size
| p | Variables | Adj. R² | AIC | BIC | Cp |
|---|---|---|---|---|---|
| 2 | PM10 | 0.352 | −2127 | −2117 | 777.4 |
| 3 | PM10, NOx | 0.443 | −2305 | −2290 | 502.7 |
| 4 | PM10, NOx, O3 | 0.497 | −2425 | −2404 | 340.5 |
| 6 | PM10, NOx + Season | 0.574 | −2618 | −2587 | 112.0 |
| 7 | PM10, NOx, O3 + Season | 0.586 | −2649 | −2614 | 77.8 |
| 8 | + NH3 | 0.595 | −2675 | −2635 | 50.8 |
| 9 | + SO2 | 0.599 | −2687 | −2642 | 38.4 |
| 10 | + NO | 0.602 | −2694 | −2643 | 32.0 |
| 11 | + NO2 | 0.607 | −2708 | −2652 | 17.4 |
| **12** | **All 8 pollutants + Season** | **0.609** | **−2714** | **−2653** | **12.0** |

### Findings
- **AIC, BIC and Mallow's Cp all choose the full model**, as does adjusted R².
- The full model has Cp = p by definition, so the real test is whether a *smaller* model also reaches the Cp = p line. None does: dropping even CO, the weakest variable, gives Cp = 17.4 against p = 11. That means the dropped variable carries real information (Figure 16).
- With n = 1181, even BIC's heavier penalty (ln n ≈ 7.1 per parameter) is outweighed by the fit each variable adds.
- **Order of importance:** PM10 → NOx → Season → O3 → NH3 → SO2 → NO → NO2 → CO. The first three variables lift adjusted R² from 0.35 to 0.57; the remaining six add only 0.04 more.

## 7b. Estimator validation: cross-validation

**Time-series cross-validation** used an expanding window with 5 folds. The first window covers one full year (316 days), so every season is seen before any prediction. Each fold then trains on all earlier days and predicts the next block of 173 days. Errors are on the µg/m³ scale; log models are back-transformed with a smearing factor estimated inside each training fold. Random 5-fold CV was run for contrast.

| Model | TS-CV RMSE (mean ± sd) | TS-CV MAE | Random 5-fold RMSE |
|---|---|---|---|
| SLR: PM10 | 13.66 ± 4.46 | 11.03 | 14.78 |
| MLR 1: pollutants | 12.48 ± 3.91 | 9.87 | 12.73 |
| MLR 2: + season | **12.29** ± 5.43 | 9.44 | 12.27 |
| log MLR 1 | 12.46 ± 3.96 | 9.59 | 14.70 |
| **log MLR 2** | 12.80 ± 5.94 | **9.36** | 12.91 |
| log MLR 2 without CO | 12.75 ± 6.17 | 9.51 | 12.94 |

### Findings
1. **Every multi-predictor model beats simple regression** (RMSE about 12.3-12.8 vs 13.7).
2. **The period being predicted matters more than the model.** Fold RMSE ranges from about 6 (calm 2019) to 20 (the winter 2016-17 spikes), while the multi-predictor models differ by under 0.5 µg/m³ on average. Cross-validation alone therefore cannot separate them.
3. **Random CV was not more optimistic here, against expectation.** Its test folds mix in the spiky 2016-17 days, while the later time-series folds are calmer. Time-series CV remains the honest method, because it never uses the future to predict the past.
4. CV error (≈ 12) is higher than the hold-out test error (≈ 8). The 2019-20 test period is calmer than the 2016-18 years inside the folds.

### Coefficient stability across folds
- **Every pollutant keeps the same sign in all 5 folds.** The direction of each effect is stable.
- The sizes vary more. PM10's coefficient rises steadily from 0.0028 (fold 1) to 0.0039 (fold 5): its link with PM2.5 strengthened over time.
- NH3, CO and NOx are the least stable, with a coefficient of variation of 0.32-0.36.

## 7c. Estimator validation: bootstrap distribution of the estimators

Two resampling schemes were used for the log MLR 2 model, each with B = 2000 resamples:
- **Case (i.i.d.) bootstrap:** resamples single days, so it assumes independence.
- **Moving-block bootstrap:** resamples blocks of 14 consecutive days, keeping the autocorrelation found in Section 6.

| Coefficient | OLS estimate | OLS SE | Block-boot SE | SE ratio | Block 95% CI | Excludes 0? |
|---|---|---|---|---|---|---|
| PM10 | 0.0042 | 0.0003 | 0.0006 | 2.27 | [0.0030, 0.0054] | ✔ |
| NO | −0.0076 | 0.0018 | 0.0028 | 1.57 | [−0.0138, −0.0030] | ✔ |
| NO2 | 0.0047 | 0.0011 | 0.0018 | 1.58 | [0.0011, 0.0084] | ✔ |
| NOx | 0.0080 | 0.0007 | 0.0013 | 1.77 | [0.0058, 0.0106] | ✔ |
| NH3 | 0.0043 | 0.0008 | 0.0024 | 3.22 | [0.0006, 0.0099] | ✔ |
| CO | −0.0283 | 0.0104 | 0.0191 | 1.83 | [−0.0551, 0.0168] | ✘ |
| SO2 | 0.0181 | 0.0047 | 0.0067 | 1.42 | [0.0054, 0.0312] | ✔ |
| O3 | 0.0039 | 0.0006 | 0.0010 | 1.57 | [0.0019, 0.0059] | ✔ |
| Monsoon | −0.363 | 0.032 | 0.068 | 2.10 | [−0.494, −0.236] | ✔ |
| Summer | −0.031 | 0.030 | 0.051 | 1.69 | [−0.127, 0.068] | ✘ |
| Post-monsoon | −0.042 | 0.032 | 0.067 | 2.09 | [−0.183, 0.076] | ✘ |

### Findings
- **Case-bootstrap SEs match the OLS SEs**, because both treat days as independent.
- **Block-bootstrap SEs are 1.4-3.2 times larger.** OLS understates uncertainty when days are dependent. This independently confirms the Newey-West result from Section 6.
- **Bias is negligible.** The bootstrap means equal the OLS estimates, so the estimators are (close to) unbiased.
- **Reliable effects:** PM10, NOx, NO, NO2, NH3, SO2, O3 and Monsoon all have block 95% intervals that exclude 0.
- **Not reliable:** CO, Summer and Post-monsoon have intervals that include 0. This matches the HAC and GLSAR results.
- **Shape of the distributions (Figure 18):** six of the eight pollutant coefficients have approximately normal sampling distributions (Shapiro p > 0.2). CO (skew 1.1) and NH3 (skew 0.6) are right-skewed, so percentile bootstrap intervals are preferred for them over normal-theory intervals.

## Selected model
**log(PM2.5) ~ PM10 + NO + NO2 + NOx + NH3 + CO + SO2 + O3 + Season** was selected for these reasons:
1. It is chosen by AIC, BIC and Mallow's Cp.
2. It has the lowest time-series CV MAE (9.36), and its RMSE is within 0.5 of the best.
3. It satisfies linearity and has near-symmetric residuals (Section 6).
4. Its coefficient signs are stable across folds, and the bootstrap confirms which effects are reliable.

CO and the Summer/Post-monsoon dummies stay in for prediction, but they are not interpreted as reliable effects.
