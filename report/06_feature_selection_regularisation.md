# Feature Selection and Regularisation

Notebook: `notebooks/AirWatch_Bengaluru_SML_AAT.ipynb` §9 (Ridge and LASSO; figure `06_ridge_lasso_paths.png`, table `regularisation_results.csv`). The feature-selection part is in the archived `archive/original_notebooks/07_feature_selection_regularisation.ipynb`.

Target: log(PM2.5). Inputs: 8 pollutants + 3 season dummies (Winter = reference). Tuning used the same 5 time-series CV folds as Section 7.

## Feature selection

Four families of methods were compared:

| Method | Type | Top features |
|---|---|---|
| Pearson correlation with log PM2.5 | Filter | Monsoon, PM10, NO2, O3, NOx |
| Mutual information | Filter (captures non-linear links) | PM10, Monsoon, NO2, NH3, NOx |
| Sequential forward selection (time-series CV) | Wrapper | Monsoon, PM10, NOx, O3, NH3 → **stops at 5** |
| LASSO coefficient path (order of drop-out) | Embedded | PM10 = Monsoon, NO2, NOx, O3 |
| All-subsets AIC/BIC/Cp (Section 7) | Information criteria | All 11 terms |

**Findings**
- **All methods agree on the core:** Monsoon, PM10 and NOx, followed by O3, NO2 and NH3. CO, Summer and Post-monsoon rank last everywhere. This matches the bootstrap result that their effects are unreliable.
- **Forward selection stops at 5 features** (log-RMSE 0.307 at 5 features, 0.332 with all 11). **This CV score is optimistic**, because the features were chosen on the same folds that score them. The result is therefore read as an *importance ranking*, not as a fixed subset.
- On the untouched test period, the 5-feature model does well (test RMSE 7.59 vs 8.05 with all 11 terms), so the weakest terms add little. AIC, BIC and Cp prefer all 11 terms; with autocorrelated data, the effective sample size is smaller than n, so in-sample criteria overstate the value of each extra variable.

## Ridge and LASSO
Features were standardised and α was chosen by time-series CV. The scaler is re-fitted inside every CV fold (a `Pipeline` in `GridSearchCV`), so no scaling information leaks between folds. This correction changed Ridge's α from 49.7 to 74.7; LASSO's α is unchanged.

| Model | α | Features used | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|---|
| OLS (log MLR 2) | — | 11 | 8.05 | 6.48 | 0.633 |
| Ridge | 74.7 | 11 | 7.92 | 6.37 | 0.645 |
| LASSO | 0.0039 | 11 | 7.90 | 6.38 | 0.646 |
| OLS on 5 forward-selected features | — | 5 | **7.59** | **6.14** | **0.673** |

- **Ridge** shrinks the weakest terms most (Summer −69%, Post-monsoon −57%, NO −21%) and barely changes PM10 (−6%) or Monsoon (−8%).
- **LASSO**, at its CV-chosen α, keeps all 11 features but pushes Summer almost to zero (−0.0008). Its coefficient path (figure `06_ridge_lasso_paths.png`) shows the order in which features would be removed as the penalty grows.
- **Regularisation gives only a small gain** over OLS (test RMSE 8.05 → 7.9), because multicollinearity is mild (all VIF < 2.5). **Choosing fewer features helps more** (7.59).
