# AirWatch Bengaluru: Statistical Modelling of PM2.5 Air Pollution

## 1. Problem Identification

### Problem statement
Fine particulate matter (PM2.5) is the air pollutant most strongly linked to respiratory and cardiovascular disease. Bengaluru's PM2.5 regularly goes above the CPCB 24-hour standard of 60 µg/m³ in winter. Yet citizens and city agencies have no simple, explainable tool that shows which pollutants drive PM2.5 or warns when a day is likely to be unhealthy.

This project builds and validates statistical models that **explain and predict daily PM2.5 in Bengaluru** from co-measured pollutants and calendar information. It also **classifies unhealthy days** using categorical data analysis and logistic regression.

### Objectives
1. Quantify how co-pollutants (PM10, NO2, NOx, O3, NH3, CO, SO2) relate to PM2.5 using simple and multiple linear regression.
2. Test the regression assumptions, diagnose outliers and influential days, and choose the best model using AIC, BIC and Mallow's Cp.
3. Model the probability of an **unhealthy day** (PM2.5 > 60 µg/m³) with logistic regression, and assess the link between season and unhealthy days with contingency-table methods.
4. Compare ordinary least squares with regularised regression (Ridge and LASSO).

### SDG alignment
| SDG | Target | How this project contributes |
|---|---|---|
| **SDG 11: Sustainable Cities and Communities** | 11.6: reduce the adverse per-capita environmental impact of cities, with special attention to air quality | Explains PM2.5 drivers and seasonality in Bengaluru to support targeted city action |
| **SDG 3: Good Health and Well-being** | 3.9: reduce deaths and illnesses from air pollution | Early warning of unhealthy days so vulnerable groups can limit exposure |

## 2. Dataset Selection

- **Source:** *Air Quality Data in India (2015-2020)*, Kaggle (Vopani/rohanrao). It is compiled from the Central Pollution Control Board (CPCB) Continuous Ambient Air Quality Monitoring network.
- **File used:** `city_day.csv` (daily city-level averages), filtered to **City = Bengaluru**.
- **Raw coverage:** 2009 consecutive days, 01-Jan-2015 to 01-Jul-2020.
- **Target variable:** `PM2.5` (µg/m³). AQI is *not* used as a target, because it is computed from the pollutants themselves; using it would leak the answer into the model.
- **Derived binary target:** `Unhealthy` = 1 if PM2.5 > 60 µg/m³ (the CPCB 24-hour National Ambient Air Quality Standard).

| Feature | Unit | Description |
|---|---|---|
| PM10 | µg/m³ | Coarse particulate matter |
| NO, NO2, NOx | µg/m³ | Nitrogen oxides (mainly vehicular) |
| NH3 | µg/m³ | Ammonia |
| CO | mg/m³ | Carbon monoxide |
| SO2 | µg/m³ | Sulphur dioxide |
| O3 | µg/m³ | Ozone |
| Benzene, Toluene | µg/m³ | Volatile organic compounds |
| Month, Season, Year | – | Calendar features (derived) |
| Lockdown | 0/1 | 25-Mar-2020 onward (COVID-19 lockdown) |

### Data cleaning
| Column | Missing % (raw) | Treatment |
|---|---|---|
| Xylene | 100.0 | Dropped |
| PM10 | 17.9 | Not recorded before 07-Nov-2015, so the sample starts on that date |
| Benzene | 13.2 | Short gaps interpolated, long gaps left missing |
| NH3 | 10.1 | Short gaps interpolated, long gaps left missing |
| PM2.5 | 7.3 | Short gaps interpolated, long gaps left missing |
| O3 | 7.2 | Short gaps interpolated, long gaps left missing |
| Others | < 5 | Short gaps interpolated |

- **Quality control (sensor errors):** two physically impossible patterns were found during the regression diagnostics (Section 6) and removed *before* interpolation:
  - **64 days with PM2.5 > PM10.** This is impossible, because PM2.5 is a subset of PM10. The worst case was 08-Jul-2017: PM2.5 = 314 with PM10 = 120. Earlier, interpolation had spread that one bad reading into a 5-day artificial spike.
  - **7 days with CO > 10 mg/m³.** Bengaluru's median is 0.9; these came in isolated bursts (Aug 2016, Feb 2017).
- Gaps of **7 days or less** were filled by time-based linear interpolation. Longer gaps were **not** filled, so no weeks of data were invented. Models use complete cases.
- **Final sample:** 1699 days (07-Nov-2015 to 01-Jul-2020); **1478 complete rows** on the 8 core pollutants.
- **Unhealthy days:** 149 of the 1649 days with a PM2.5 reading (9.0%). The classes are imbalanced, which the logistic-regression section handles.

**Limitation:** interpolation smooths short gaps and slightly understates day-to-day variance. Results are city-level averages, not station-level exposure.

## 3. Exploratory Data Analysis

Notebook: `notebooks/AirWatch_Bengaluru_SML_AAT.ipynb` §2-3 · Figure `reports/figures/01_eda.png` (the histograms and scatter plots are archived in `archive/original_outputs/`).

### Descriptive statistics (PM2.5)
| n | Mean | SD | Median | Min | Max | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|
| 1649 | 35.2 | 18.1 | 31.7 | 1.7 | 204.3 | 1.84 | 8.8 |

### PM2.5 by season
| Season | Mean | Median | Max |
|---|---|---|---|
| Winter (Dec-Feb) | 45.6 | 42.3 | 204.2 |
| Summer (Mar-May) | 38.3 | 35.7 | 187.5 |
| Monsoon (Jun-Sep) | 21.9 | 20.7 | 75.4 |
| Post-monsoon (Oct-Nov) | 37.6 | 34.9 | 102.0 |

### Correlation of PM2.5 with other pollutants
PM10 0.62 · NO2 0.48 · O3 0.39 · NOx 0.33 · NO 0.19 · SO2 0.19 · Benzene 0.18 · NH3 0.12 · CO 0.08 · Toluene 0.01

### Key observations
1. **Right-skewed target.** PM2.5 has skew 1.8 and a tail of spike days above 100 µg/m³. Expect non-normal residuals, so a log transform will be considered.
2. **Strong seasonality.** Winter mean (45.6) is about 2× the monsoon mean (21.9), because rain washes particulates out and the winter boundary layer traps them.
3. **Shared sources.** PM2.5 correlates most with PM10 (0.62) and NO2 (0.48), which points to road dust and traffic.
4. **Multicollinearity risk.** NO, NO2 and NOx are chemically linked, so VIF will be checked before interpretation.
5. **Persistence over time.** Daily PM2.5 is strongly autocorrelated (lag-1 r = 0.76): polluted days follow polluted days. Expect a low Durbin-Watson statistic.
6. **Lockdown.** Mean PM2.5 was 21.7 µg/m³ during the 2020 COVID lockdown against 36.1 otherwise (−40%). This raw comparison does not adjust for season.
7. **Year-to-year.** 2016 was the most polluted year (mean 43.5). From 2017 to 2020 the annual mean stayed between 30 and 35.
