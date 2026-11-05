# Results: Summary of Statistical Findings

This section brings together Sections 4-7 and the categorical analysis. Details and tables are in `02_regression.md`, `03_diagnostics.md`, `04_validation.md` and `05_logistic.md`.

## Final regression model
**log(PM2.5) = 2.714 − 0.363·Monsoon − 0.031·Summer − 0.042·Post-monsoon + 0.0042·PM10 − 0.0076·NO + 0.0047·NO2 + 0.0080·NOx + 0.0043·NH3 − 0.0283·CO + 0.0181·SO2 + 0.0039·O3**

It was fitted on 1181 training days (Nov 2015 to Sep 2019), with adjusted R² = 0.609 and test RMSE = 8.05 µg/m³ (test R² = 0.633).

## Interpretation of the coefficients
Each effect below holds the other variables fixed. Only effects that stay significant under the HAC, GLSAR **and** block-bootstrap checks are interpreted.

| Factor | Effect on PM2.5 | Meaning |
|---|---|---|
| Monsoon (vs winter) | **−30%** | Rain washes particles out of the air |
| NOx, +10 µg/m³ | **+8.3%** | Vehicle exhaust, and a precursor of secondary particles |
| NO2, +10 µg/m³ | +4.9% | Traffic combustion |
| NH3, +10 µg/m³ | +4.4% | Waste and agriculture; forms ammonium salts |
| PM10, +10 µg/m³ | **+4.3%** | Road and construction dust share sources with PM2.5 |
| O3, +10 µg/m³ | +4.0% | Photochemical activity, which also forms secondary particles |
| SO2, +10 µg/m³ | +19.8% | Large per unit, but SO2 is low and varies little in Bengaluru |
| NO, +10 µg/m³ | −7.3% | Reflects the overlap NOx ≈ NO + NO2, not a cleaning effect |
| CO, Summer, Post-monsoon | not reliable | Their confidence intervals include 0 under the robust methods |

## Model performance
| Stage | Model | Adjusted R² | Test RMSE (µg/m³) |
|---|---|---|---|
| Simple regression | PM2.5 ~ PM10 | 0.362 | 8.08 |
| Multiple regression | + 7 pollutants + season | 0.563 | 8.57 |
| Remedied model | log(PM2.5), same terms | **0.609** | **8.05** |

## Key statistical findings
1. **PM10 alone explains 36% of daily PM2.5 variation** (r = 0.62). Adding traffic gases and season raises this to 56-61%.
2. **Assumptions:** the linear model violated linearity, normality, constant variance and independence. The log transform fixed linearity and most of the skewness. Robust (HC3, Newey-West) standard errors and Cochrane-Orcutt GLS handled heteroscedasticity and autocorrelation (Durbin-Watson 0.90 → 2.07). Multicollinearity was not an issue (VIF < 2.5).
3. **Diagnostics improved the data.** Cook's distance exposed sensor errors (PM2.5 > PM10), which were removed in cleaning. After the fix, no day is strongly influential (maximum Cook's D = 0.056).
4. **Model selection:** AIC, BIC and Mallow's Cp all choose the full model; in importance order, PM10, NOx and Season matter most.
5. **Estimator validation:**
   - Coefficient signs are stable across all CV folds.
   - The block bootstrap shows that OLS standard errors are 1.4-3.2 times too small when the day-to-day dependence is ignored.
   - Bias is negligible.
6. **Unhealthy days** (PM2.5 > 60) depend strongly on season (χ² = 111.4, p < 10⁻²³); the odds are 4.4 times higher in winter. Logistic regression identifies PM10, NOx, winter, SO2 and NO2 as the drivers. It has AUC 0.89 and catches 82% of unhealthy days at a 0.12 threshold.
7. **PM2.5 averaged about 40% lower during the 2020 COVID lockdown** (21.7 vs 36.1 µg/m³). This raw comparison does not adjust for season, so it is consistent with human activity driving PM2.5 but does not prove it.

## Limitations
- These are associations in observational, city-level data, so they do not prove cause and effect.
- The NO2 effect depends on the period: +6.5% per 10 µg/m³ over the full period, but +0.6% and not significant in 2018-20.
- The test period is unusually clean and includes the lockdown, with only 7 unhealthy days. At the 0.12 threshold the logistic model caught all 7 but raised 44 false alarms, so only 14% of its alerts were correct.
