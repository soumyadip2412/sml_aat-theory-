# Categorical Data Analysis and Logistic Regression

Notebook: `notebooks/06_categorical_logistic.ipynb` · Figures 19-22 · Tables `contingency_*`, `odds_ratios.csv`, `logistic_*.csv`

**Question:** what makes a day **unhealthy** (PM2.5 > 60 µg/m³, the CPCB 24-hour standard), and can it be predicted from the other pollutants? 149 of the 1649 days with a PM2.5 reading (9.0%) were unhealthy.

## Contingency table: season × air quality

| Season | Acceptable | Unhealthy | % unhealthy | Expected unhealthy under H₀ |
|---|---|---|---|---|
| Winter | 355 | 86 | **19.5** | 39.8 |
| Post-monsoon | 236 | 32 | 11.9 | 24.2 |
| Summer | 414 | 30 | 6.8 | 40.1 |
| Monsoon | 495 | 1 | **0.2** | 44.8 |

**Chi-squared test of independence:** χ² = 111.42, df = 3, p = 5 × 10⁻²⁴. The smallest expected count is 24.2, well above 5, so the test is valid. **Air quality depends strongly on season.** Cramér's V = 0.26, a moderate association.

The Pearson residuals (Figure 19) show where the dependence comes from: winter has more than twice the unhealthy days expected under independence, and the monsoon has almost none (1 day against 45 expected).

## Odds ratios

| Comparison | Odds (group) | Odds (other) | Odds ratio | 95% CI | p (Fisher) |
|---|---|---|---|---|---|
| Winter vs other seasons | 0.242 | 0.055 | **4.40** | [3.11, 6.22] | < 0.001 |
| Weekday vs weekend | 0.103 | 0.090 | 1.15 | [0.79, 1.69] | 0.51 |

- **The odds of an unhealthy day are 4.4 times higher in winter** than in the rest of the year (risk ratio 3.7).
- **Weekdays are not significantly worse than weekends.** The confidence interval includes 1. Daily traffic changes within a week do not shift the share of days that cross the standard; seasonal weather matters far more.

## Logistic regression

**Model:** logit(P(unhealthy)) = β₀ + β·(8 pollutants) + β_W·Winter. It was fitted by maximum likelihood on the 1181 training days, of which 127 (10.8%) were unhealthy.

Season enters as a single Winter indicator. The monsoon has only 1 unhealthy day, so a separate Monsoon dummy would be almost perfectly separated and could not be estimated.

### Overall fit
- **Likelihood-ratio test:** G² = 296.6, df = 9, p ≈ 10⁻⁵⁸. The model is far better than the intercept-only model.
- **McFadden pseudo-R²** = 0.368. Values of 0.2-0.4 indicate a good fit for logistic models.
- **Hosmer-Lemeshow test:** χ² = 4.38, df = 8, p = 0.82. There is no evidence of poor calibration: predicted and observed rates match in every probability decile.

### Odds ratios (per 10 µg/m³; CO per 1 mg/m³)
| Variable | Odds ratio | 95% CI | Wald p | Drop-one LR G² (p) |
|---|---|---|---|---|
| PM10 | **1.34** | [1.25, 1.43] | < 0.001 | 89.5 (< 0.001) |
| NOx | **1.55** | [1.30, 1.84] | < 0.001 | 29.1 (< 0.001) |
| Winter (vs other seasons) | **2.30** | [1.35, 3.90] | 0.002 | 9.5 (0.002) |
| SO2 | **4.08** | [1.44, 11.56] | 0.008 | 7.8 (0.005) |
| NO2 | **1.30** | [1.02, 1.66] | 0.034 | 4.5 (0.033) |
| O3 | 1.14 | [0.99, 1.31] | 0.073 | 3.2 (0.074) |
| NH3 | 1.10 | [0.93, 1.29] | 0.279 | 1.1 (0.294) |
| NO | 0.90 | [0.58, 1.39] | 0.637 | 0.2 (0.634) |
| CO | 1.05 | [0.80, 1.37] | 0.730 | 0.1 (0.737) |

**Interpretation (holding the other variables fixed):**
- Each 10 µg/m³ rise in **PM10** multiplies the odds of an unhealthy day by 1.34, a 34% increase.
- Each 10 µg/m³ rise in **NOx** raises the odds by 55%.
- **Winter** more than doubles the odds (OR 2.30) even after accounting for the pollutant levels. Winter weather traps particles beyond what the co-pollutants explain. The raw winter odds ratio of 4.4 shrinks to 2.3 once the pollutants are controlled for, because part of the winter effect works through higher pollutant levels.
- The **SO2** odds ratio is large per 10 µg/m³, but SO2 rarely varies by that much (median 5 µg/m³), and its confidence interval is wide.
- **NH3, NO and CO add nothing** once the others are known.

The conclusion matches the linear models: PM10 and NOx drive unhealthy days, and winter amplifies them (Figure 20).

## Predictive power: sensitivity, specificity and ROC

| Data / threshold | Sensitivity | Specificity | Precision | Accuracy | AUC |
|---|---|---|---|---|---|
| Train @ 0.50 | 0.370 | 0.988 | 0.783 | 0.921 | 0.894 |
| **Train @ 0.12 (Youden)** | **0.819** | **0.814** | 0.347 | 0.815 | 0.894 |
| Test @ 0.50 | 0.286 (2/7) | 0.990 | 0.400 | 0.973 | 0.968 |
| **Test @ 0.12 (Youden)** | **1.000 (7/7)** | **0.848** | 0.137 | 0.852 | 0.968 |

- **AUC = 0.89 on training data and 0.97 on test data.** The model ranks unhealthy days above acceptable days very well.
- **The default 0.5 threshold is the wrong choice here.** It catches only 37% of the unhealthy training days, because they are rare (11%). Its high accuracy (92%) is misleading: a model that always said "acceptable" would score 89%.
- **The threshold was chosen on the training data with Youden's J**, which maximises sensitivity + specificity − 1. At 0.12, the model catches 82% of unhealthy days while still clearing 81% of acceptable ones. For a public-health warning, missing an unhealthy day costs more than a false alarm, so this trade-off is the right one (Figure 22).
- **On the test period, all 7 unhealthy days were flagged**, with 85% specificity. With only 7 positive days the test sensitivity is very uncertain (an exact 95% interval runs from about 59% to 100%). The training-set figures are the more reliable estimate.
- **Precision is low (14-35%):** most warnings are false alarms. That is the unavoidable cost of high sensitivity when the event is rare.
