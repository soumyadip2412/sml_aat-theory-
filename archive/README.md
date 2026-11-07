# Archive

The submission notebook is `notebooks/AirWatch_Bengaluru_SML_AAT.ipynb`. Everything here is the earlier version, superseded by that notebook and kept for reference.

- `original_notebooks/`: the original analysis notebooks 01-07. They contain extras beyond the syllabus (cross-validation, bootstrap, feature selection, calibration). They expect to be run from `notebooks/`, so move them back there if you want to re-run them.
- `original_outputs/`: the figures and tables those notebooks produced, including the original versions of tables that the new notebook now regenerates.

Two methodological corrections were made in the new notebook:
1. The added-variable plots now come from the full log model with season. The originals used a model without season.
2. Ridge/LASSO α is tuned with the scaler re-fitted inside each CV fold. Ridge α changed from 49.7 to 74.7 (test RMSE 7.94 → 7.92); LASSO is unchanged.
