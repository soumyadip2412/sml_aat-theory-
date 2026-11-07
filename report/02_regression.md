# Simple and Multiple Linear Regression

Notebook: `notebooks/AirWatch_Bengaluru_SML_AAT.ipynb` §4-5 · Figures 02 · Tables `slr_*`, `mlr_*`, `regression_comparison.csv`

## Modelling set-up
- **Data:** 1478 days with complete values for PM2.5 and 8 co-pollutants (PM10, NO, NO2, NOx, NH3, CO, SO2, O3), after quality control.
- **Train/test split:** chronological 80/20, with no shuffling because the data is a time series.
  - Train: 1181 days (07-Nov-2015 to 08-Sep-2019). All tests, intervals and ANOVA use this set.
  - Test: 297 days (09-Sep-2019 to 01-Jul-2020). Held out for honest evaluation.

## 4. Simple Linear Regression

**Model:** PM2.5 = β₀ + β₁·PM10 + ε. PM10 was chosen because it has the strongest correlation with PM2.5 (r = 0.62).

### Least-squares estimates
The estimates were first computed by hand (β̂₁ = Sxy/Sxx, β̂₀ = ȳ − β̂₁x̄) and then confirmed with statsmodels.

| Parameter | Estimate | Std. error | t | p-value | 95% CI |
|---|---|---|---|---|---|
| β₀ (intercept) | 10.994 | 1.068 | 10.29 | < 0.001 | [8.90, 13.09] |
| β₁ (PM10) | 0.280 | 0.011 | 25.89 | < 0.001 | [0.259, 0.302] |

**Fitted line:** PM2.5 = 10.99 + 0.280 × PM10, with σ̂ = 14.86 µg/m³.

### Hypothesis test
H₀: β₁ = 0 vs H₁: β₁ ≠ 0. The test gives t = 25.89, far above t₀.₀₂₅,₁₁₇₉ = 1.962 (p ≈ 2 × 10⁻¹¹⁷). H₀ is rejected: PM10 has a significant positive linear effect on PM2.5.

**Interpretation:** each 10 µg/m³ rise in PM10 is associated with a 2.8 µg/m³ rise in PM2.5 (95% CI: 2.6 to 3.0).

### ANOVA
| Source | df | SS | MS | F | p |
|---|---|---|---|---|---|
| Regression | 1 | 148,057 | 148,057 | 670.36 | < 0.001 |
| Residual | 1179 | 260,399 | 220.9 | | |
| Total | 1180 | 408,456 | | | |

As expected in simple regression, F = t² (670.36 = 25.89²).

### Coefficient of determination
R² = 0.3625 and adjusted R² = 0.3619. PM10 alone explains about 36% of the day-to-day variation in PM2.5. R² equals the squared correlation, as theory predicts for simple regression.

### Prediction of new observations
| PM10 | Predicted PM2.5 | 95% CI (mean) | 95% prediction interval (one day) |
|---|---|---|---|
| 50 | 25.0 | [23.8, 26.2] | [−4.2, 54.2] |
| 100 | 39.0 | [38.2, 39.9] | [9.9, 68.2] |
| 150 | 53.1 | [51.5, 54.6] | [23.9, 82.2] |

The prediction interval is much wider than the confidence interval, because it adds the irreducible day-to-day noise (σ²). At low PM10 its lower limit goes **negative**, which is physically impossible. This is a sign that the error variance is not constant, and Section 6 addresses it with a log transform.

## 5. Multiple Linear Regression

In matrix form, β̂ = (XᵀX)⁻¹Xᵀy. The manual matrix calculation matched statsmodels exactly. By the **Gauss-Markov theorem**, OLS is the Best Linear Unbiased Estimator *if* the errors have zero mean, constant variance and no correlation. These conditions are tested in Section 6.

Two models were fitted:
- **Model 1:** PM2.5 on all 8 pollutants.
- **Model 2:** Model 1 plus Season dummies, with Winter as the reference.

The lockdown indicator could not be used, because every lockdown day falls in the test period.

### Model 2 coefficients
| Term | Estimate | p-value | Significant at 5% |
|---|---|---|---|
| Intercept | 6.725 | 0.004 | ✔ |
| Season: Monsoon (vs Winter) | −10.925 | < 0.001 | ✔ |
| Season: Summer | −2.343 | 0.046 | ✔ |
| Season: Post-monsoon | −2.305 | 0.066 | |
| PM10 | 0.180 | < 0.001 | ✔ |
| NOx | 0.304 | < 0.001 | ✔ |
| NO2 | 0.164 | < 0.001 | ✔ |
| NO | −0.274 | < 0.001 | ✔ |
| NH3 | 0.123 | < 0.001 | ✔ |
| SO2 | 0.830 | < 0.001 | ✔ |
| O3 | 0.101 | < 0.001 | ✔ |
| CO | −0.598 | 0.141 | |

**Interpretation (holding the other variables constant):**
- **Monsoon** days average **10.9 µg/m³ lower** PM2.5 than winter days. This matches rain wash-out of particulates. Summer is about 2.3 lower.
- A 10 µg/m³ rise in **PM10** adds 1.8 µg/m³ of PM2.5. This is smaller than in the simple model (2.8), because part of PM10's apparent effect was really shared with traffic gases and season.
- **NOx** and **NO2** (vehicle exhaust) and **NH3** (agriculture and waste) all raise PM2.5. All are known precursors of secondary particulate formation.
- **NO has a negative coefficient.** NO, NO2 and NOx measure overlapping chemistry (NOx ≈ NO + NO2): holding NOx fixed, more NO means less of the other oxides. VIF (Section 6) confirms the overlap is not severe, but these three coefficients should not be interpreted one at a time.
- **SO2** has a large per-unit effect (0.83), but its range is small (median 5 µg/m³), so its practical impact is modest.

### ANOVA tests
- **Overall F-test (Model 2):** F = 139.2 (p ≈ 10⁻²⁰³). At least one predictor is related to PM2.5.
- **Partial F-test (Model 1 vs Model 2):** F = 31.94 on 3 df (p < 0.001). Adding season significantly improves the fit.
- **Type II ANOVA:** PM10 (F = 268.6) and NOx (116.3) carry most of the explanatory power, followed by Season (31.9).

## Model comparison
| Model | Params | R² (train) | Adj. R² (train) | AIC | BIC | RMSE (test) | MAE (test) | R² (test) |
|---|---|---|---|---|---|---|---|---|
| SLR: PM10 | 2 | 0.363 | 0.362 | 9728 | 9738 | 8.08 | 6.08 | 0.630 |
| MLR 1: pollutants | 9 | 0.532 | 0.528 | 9378 | 9424 | **7.02** | **5.48** | **0.721** |
| MLR 2: + season | 12 | **0.567** | **0.563** | **9291** | **9352** | 8.57 | 6.86 | 0.584 |

### Findings
1. Multiple regression clearly beats simple regression. Adjusted R² rises from 0.36 to 0.56, and AIC and BIC both fall sharply.
2. **Model 2 fits the training data best, but Model 1 predicts the test period better.** The seasonal effects learned from 2015-2019 did not carry over to late-2019/2020, when PM2.5 was unusually low (including the COVID lockdown). This is a real-world sign of *distribution shift*, and it motivates cross-validation in the validation section.
3. Test R² is higher than train R². The training period contains more extreme spike days, which inflate the error, while the test period is calmer.
4. The residuals are still right-skewed (skew ≈ 2.6) and autocorrelated (Durbin-Watson ≈ 1.0). This means the OLS assumptions are violated; the problem is diagnosed and remedied in Section 6.
