"""Refresh the numbers inside the frontend pages from the project notebook.

Re-runs the code cells of notebooks/AirWatch_Bengaluru_SML_AAT.ipynb in memory (nothing in
data/ or reports/ is written), collects every result the pages show, adds the notebook's own
code for the walkthrough's code cards, and replaces the data block in frontend/index.html and
frontend/walkthrough.html.

Run after changing the notebook:   venv\\Scripts\\python frontend\\build_data.py
"""
import io, json, re, sys, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "AirWatch_Bengaluru_SML_AAT.ipynb"
PAGES = [ROOT / "frontend" / "index.html", ROOT / "frontend" / "walkthrough.html"]

# Walkthrough steps: file name shown on the code card, and the first line of each notebook cell it shows
CODE_STEPS = [
    ("01_setup.py", ["# 1. Setup"]),
    ("02_clean_and_split.py", ["# 2. Load, clean and split"]),
    ("03_eda.py", ["# 3. EDA"]),
    ("04_simple_regression.py", ["x, y = train", "# SLR prediction"]),
    ("05_multiple_regression.py", ["# 5. MLR", "# Out-of-sample check"]),
    ("06_assumptions.py", ["# 6. Assumption checks"]),
    ("07_influence.py", ["# 7. Added-variable plots", "# Influence diagnostics"]),
    ("08_selection_and_shrinkage.py", ["# 8. All-subsets selection", "# 9. Ridge and LASSO"]),
    ("09_seasons_and_logistic.py", ["# 10. Contingency table", "# Logistic regression", "# Classification performance"]),
    ("10_final_comparison.py", ["# 11. Final comparison"]),
]


def run_notebook(cells):
    """Execute the notebook's code cells and return its variables."""
    ns = {"display": lambda *a, **k: None}
    keep = pd.DataFrame.to_csv, pd.Series.to_csv, plt.savefig, plt.show
    pd.DataFrame.to_csv = pd.Series.to_csv = plt.savefig = plt.show = lambda *a, **k: None
    out, sys.stdout = sys.stdout, io.StringIO()
    try:
        for c in cells:
            if c["cell_type"] == "code":
                src = "".join(c["source"]).replace("../data/raw/", (ROOT / "data" / "raw").as_posix() + "/")
                exec(compile(src, "notebook", "exec"), ns)
                plt.close("all")
    finally:
        sys.stdout = out
        pd.DataFrame.to_csv, pd.Series.to_csv, plt.savefig, plt.show = keep
    return ns


def r(a, k=3):
    """Round an array-like to k decimals; NaN becomes null."""
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(v) else round(float(v), k) for v in a.ravel()]


def code_steps(cells):
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    steps = []
    for name, starts in CODE_STEPS:
        parts = []
        for s in starts:
            hit = [c for c in code if c.lstrip().startswith(s)]
            if len(hit) != 1:
                sys.exit(f"code step {name}: expected one cell starting with {s!r}, found {len(hit)}")
            parts.append(hit[0].rstrip())
        steps.append(dict(file=name, code="\n\n".join(parts)))
    return steps


def collect(ns):
    df, data, train, test = ns["df"], ns["data"], ns["train"], ns["test"]
    X_VARS, ORDER, n = ns["X_VARS"], ns["ORDER"], ns["n"]
    slr, mlr1, mlr2, log2, logit = ns["slr"], ns["mlr1"], ns["mlr2"], ns["log2"], ns["logit"]
    short = lambda s: re.sub(r"C\(Season.*?\)\[T\.(.*?)\]", r"\1", s)
    D = {}

    raw = pd.read_csv(ROOT / "data" / "raw" / "city_day.csv")
    blr = raw[raw["City"] == "Bengaluru"]
    D["meta"] = dict(days=len(df), start=str(df["Date"].min().date()), end=str(df["Date"].max().date()),
                     bad_pm=int(ns["bad_pm"].sum()), bad_co=int(ns["bad_co"].sum()), complete=len(data),
                     n_train=n, n_test=len(test), train_end=str(train["Date"].max().date()),
                     test_start=str(test["Date"].min().date()), lockdown_start="2020-03-25",
                     unhealthy_train=int(train["Unhealthy"].sum()), unhealthy_test=int(test["Unhealthy"].sum()),
                     raw_rows=len(blr), raw_cols=blr.shape[1], raw_numeric=int(blr.select_dtypes("number").shape[1]),
                     raw_text=int(blr.select_dtypes("object").shape[1]) - 1, raw_date=1)

    # EDA
    day0 = df["Date"].min()
    seas = df.groupby("Season")["PM2.5"].mean().reindex(ORDER)
    corr = df[["PM2.5"] + X_VARS].corr()
    D["eda"] = dict(day=((df["Date"] - day0).dt.days).tolist(), pm25=r(df["PM2.5"], 1),
                    roll=r(df["PM2.5"].rolling(30, center=True).mean(), 1),
                    monthly=r(df.groupby("Month")["PM2.5"].mean(), 1),
                    season=dict(zip(ORDER, r(seas, 1))), corr_names=list(corr.columns), corr=[r(row, 2) for row in corr.values])

    # Simple linear regression
    x, y = train["PM10"].values, train["PM25"].values
    ci = slr.conf_int()
    D["slr"] = dict(x=r(x, 1), y=r(y, 1), b0=float(slr.params.iloc[0]), b1=float(slr.params.iloc[1]),
                    se=r(slr.bse, 4), t=r(slr.tvalues, 2), p=[float(v) for v in slr.pvalues], ci=[r(ci.iloc[0], 3), r(ci.iloc[1], 3)],
                    r2=float(slr.rsquared), adj=float(slr.rsquared_adj), F=float(slr.fvalue), n=n,
                    xbar=float(x.mean()), Sxx=float(((x - x.mean()) ** 2).sum()), mse=float(slr.mse_resid),
                    tcrit=float(stats.t.ppf(0.975, n - 2)), ssr=float(slr.ess), sse=float(slr.ssr), sst=float(slr.centered_tss))

    # Multiple linear regression
    partial = ns["partial"]
    def scores(pred):
        rm, r2 = ns["test_scores"](pred)
        return [float(rm), float(r2)]
    D["mlr"] = dict(terms=[short(s) for s in mlr2.params.index], coef=r(mlr2.params, 4), se=r(mlr2.bse, 4), t=r(mlr2.tvalues, 2),
                    p=[float(v) for v in mlr2.pvalues], F=float(mlr2.fvalue), Fp=float(mlr2.f_pvalue),
                    partialF=float(partial["F"].iloc[1]), partialP=float(partial["Pr(>F)"].iloc[1]),
                    perf=[dict(name=nm, r2=float(m.rsquared), adj=float(m.rsquared_adj), rmse=scores(m.predict(test))[0], r2test=scores(m.predict(test))[1])
                          for nm, m in [("SLR: PM10", slr), ("MLR 1: 8 pollutants", mlr1), ("MLR 2: + season", mlr2)]])

    # Assumption checks
    def resid_block(m):
        res = np.sort((m.resid - m.resid.mean()) / m.resid.std())
        q = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
        return dict(fitted=r(m.fittedvalues, 3), resid=r(m.resid, 3), qq_t=r(q, 3), qq_s=r(res, 3))
    tests = ns["tests"]
    D["diag"] = dict(raw=resid_block(mlr2), log=resid_block(log2), vif=dict(zip(X_VARS, r(ns["vif"], 2))),
                     tests=dict(rows=list(tests.index), raw=r(tests.iloc[:, 0], 4), log=r(tests.iloc[:, 1], 4)),
                     rho=float(ns["glsar"].model.rho[0]), hac_sig=int((ns["hac"].pvalues[1:] < 0.05).sum()), hac_n=len(ns["hac"].pvalues) - 1)

    # Influence diagnostics and added-variable plots
    meas, it = ns["measures"], ns["infl_tab"]
    X, yy, names = log2.model.exog, log2.model.endog, list(log2.model.exog_names)
    av = []
    for v in X_VARS:
        k = names.index(v); others = np.delete(X, k, axis=1)
        ey = yy - others @ np.linalg.lstsq(others, yy, rcond=None)[0]
        ex = X[:, k] - others @ np.linalg.lstsq(others, X[:, k], rcond=None)[0]
        av.append(dict(name=v, x=r(ex, 2), y=r(ey, 3), slope=float(log2.params[v]), t=float(log2.tvalues[v])))
    top = int(meas["Cook's D"].idxmax())
    D["infl"] = dict(day=((train["Date"] - day0).dt.days).tolist(), names=list(meas.columns), values=[r(meas[c], 4) for c in meas], cut=r(it["cut-off"], 4),
                     rules=list(it["rule"]), labels=list(it.index), flagged=[int(v) for v in it["days flagged"]],
                     top=dict(date=str(train.loc[top, "Date"].date()), pm25=float(train.loc[top, "PM25"]), d=float(meas["Cook's D"].max())), av=av)

    # Model selection
    sub, best = ns["subsets"], ns["best"]
    D["select"] = dict(p=[int(v) for v in sub["p"]], aic=r(sub["AIC"], 1), bic=r(sub["BIC"], 1), cp=r(sub["Cp"], 1), adj=r(sub["adj R²"], 3),
                       model=list(sub["model"]),
                       best=[dict(label=i, model=row["model"], p=int(row["p"]), adj=float(row["adj R²"]), aic=float(row["AIC"]),
                                  bic=float(row["BIC"]), cp=float(row["Cp"])) for i, row in best.iterrows()])

    # Ridge and LASSO
    reg, models = ns["reg"], ns["models"]
    feats = list(ns["X_tr"].columns)
    D["reg"] = dict(features=feats, coef={c: r(reg.loc[feats, c], 4) for c in reg.columns},
                    alpha=dict(Ridge=float(models["Ridge"][-1].alpha), LASSO=float(models["LASSO"][-1].alpha)),
                    nonzero={c: int(reg.loc["non-zero coefficients", c]) for c in reg.columns},
                    rmse={c: float(reg.loc["RMSE test", c]) for c in reg.columns}, r2={c: float(reg.loc["R² test", c]) for c in reg.columns},
                    ridge_alpha=r(np.log10(ns["alphas_r"]), 3), ridge_path=[r(row, 4) for row in ns["ridge_path"].T],
                    lasso_alpha=r(np.log10(ns["alphas_l"]), 3), lasso_path=[r(row, 4) for row in ns["lasso_coefs"]])

    # Contingency table (counts read from the table itself, since later cells reuse the names a, b, c, d)
    obs = ns["obs"]
    a, b = int(obs.loc["Winter", "Unhealthy"]), int(obs.loc["Winter", "Acceptable"])
    c, d = int(obs.drop("Winter")["Unhealthy"].sum()), int(obs.drop("Winter")["Acceptable"].sum())
    D["cat"] = dict(seasons=ORDER, acceptable=[int(v) for v in obs["Acceptable"]], unhealthy=[int(v) for v in obs["Unhealthy"]],
                    expected=[r(row, 1) for row in ns["expected"]], chi2=float(ns["chi2"]), dof=int(ns["dof"]), p=float(ns["p_chi"]),
                    v=float(ns["cramers_v"]), odds=float(ns["odds_ratio"]),
                    ci=[float(np.exp(np.log(ns["odds_ratio"]) + s * 1.96 * ns["se_log"])) for s in (-1, 1)], abcd=[a, b, c, d])

    # Logistic regression
    ot, clf = ns["or_tab"], ns["clf"]
    D["logit"] = dict(terms=list(ot.index), coef=r(ot["coef (log-odds)"], 4), p=[float(v) for v in ot["p (Wald)"]], unit=r(ot["per unit"], 0),
                      odds=r(ot["odds ratio"], 3), lo=r(ot["OR 95% CI low"], 3), hi=r(ot["OR 95% CI high"], 3),
                      params={k: float(v) for k, v in logit.params.items()}, G2=float(logit.llr), df=int(logit.df_model),
                      lrp=float(logit.llr_pvalue), mcfadden=float(logit.prsquared), cutoff=float(ns["cutoff"]),
                      p_train=r(ns["p_tr"], 6), y_train=[int(v) for v in train["Unhealthy"]],
                      p_test=r(ns["p_te"], 6), y_test=[int(v) for v in test["Unhealthy"]],
                      auc=dict(train=float(clf.loc["Train", "AUC"]), test=float(clf.loc["Test", "AUC"])))

    # Final model: coefficients for the live predictor, test-period predictions, final table
    final = ns["final"]
    pm = ns["pm25_pred"]
    preds = {"SLR: PM10": slr.predict(test), "MLR 2: pollutants + season": mlr2.predict(test), "Final: log MLR 2": pm(models["OLS"]),
             "Ridge (log)": pm(models["Ridge"]), "LASSO (log)": pm(models["LASSO"])}
    med = train.groupby("Season")[X_VARS].median().reindex(ORDER)
    D["final"] = dict(table=[dict(name=i, adj=None if pd.isna(row.iloc[0]) else float(row.iloc[0]), rmse=float(row.iloc[1]), r2=float(row.iloc[2]))
                             for i, row in final.iterrows()],
                      terms=[short(s) for s in log2.params.index], beta=[float(v) for v in log2.params],
                      cov=[[float(v) for v in row] for row in log2.cov_params().values], s2=float(log2.mse_resid),
                      tcrit=float(stats.t.ppf(0.975, log2.df_resid)), smear=float(np.mean(np.exp(log2.resid))), adj=float(log2.rsquared_adj),
                      vars=X_VARS, seasons=ORDER,
                      range={v: [float(np.floor(data[v].min())), float(np.ceil(data[v].quantile(0.995)))] for v in X_VARS},
                      median={s: r(med.loc[s], 1) for s in ORDER},
                      test=dict(day=((test["Date"] - day0).dt.days).tolist(), date=[str(t.date()) for t in test["Date"]],
                                actual=r(test["PM25"], 1), pred=r(preds["Final: log MLR 2"], 1), season=list(test["Season"]),
                                x=[r(test[v], 2) for v in X_VARS], preds={k: r(v, 2) for k, v in preds.items()}))
    return D


def improve(ns):
    """Follow-up experiments for the 'How we could improve it' section (same split, train-only fitting)."""
    from sklearn.metrics import roc_auc_score, roc_curve
    from statsmodels.stats.stattools import durbin_watson
    smf, X_VARS, RHS, SEASON, split = ns["smf"], ns["X_VARS"], ns["RHS"], ns["SEASON"], ns["split_date"]
    d = ns["data"].sort_values("Date").reset_index(drop=True)
    prev = d["Date"].diff().dt.days == 1                      # yesterday must be the calendar day before
    d["PM25_lag1"] = d["PM25"].shift(1).where(prev)
    d["logPM25_lag1"] = np.log(d["PM25_lag1"])
    for c in X_VARS:
        d["l" + c] = np.log1p(d[c])
    d = d.dropna(subset=["PM25_lag1"])                         # same rows for every model, so the comparison is fair
    tr, te = d[d["Date"] < split], d[d["Date"] >= split]

    def score(y, p):
        e = np.asarray(y) - np.asarray(p)
        return dict(rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()), bias=float(e.mean()),
                    r2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()))

    def cv_rmse(formula, log):                                 # expanding-window CV inside the training period
        k, first = len(tr), int((tr["Date"] < tr["Date"].min() + pd.DateOffset(years=1)).sum())
        size, out = (k - first) // 5, []
        for i in range(5):
            a = k - (5 - i) * size
            fit, val = tr.iloc[:a], tr.iloc[a:a + size]
            m = smf.ols(formula, data=fit).fit(); p = m.predict(val)
            p = np.exp(p) * np.mean(np.exp(m.resid)) if log else p
            out.append(score(val["PM25"].values, p)["rmse"])
        return float(np.mean(out))

    LP = " + ".join("l" + c for c in X_VARS)
    specs = [("Notebook model: log MLR 2", "np.log(PM25) ~ " + RHS, True),
             ("8 pollutants, no season (MLR 1)", "PM25 ~ " + " + ".join(X_VARS), False),
             ("Log model + PM10 × season", "np.log(PM25) ~ " + LP + " + " + SEASON + " + lPM10:" + SEASON, True),
             ("Notebook model + yesterday's PM2.5", "np.log(PM25) ~ " + RHS + " + logPM25_lag1", True)]
    rows, preds = [], {}
    for name, fml, log in specs:
        m = smf.ols(fml, data=tr).fit(); p = m.predict(te)
        p = np.exp(p) * np.mean(np.exp(m.resid)) if log else p
        preds[name] = p
        rows.append(dict(name=name, cv=cv_rmse(fml, log), adj=float(m.rsquared_adj), dw=float(durbin_watson(m.resid)), **score(te["PM25"].values, p)))
    rows.insert(1, dict(name="No model: same as yesterday", cv=None, adj=None, dw=None, **score(te["PM25"].values, te["PM25_lag1"].values)))

    def logi(fml):
        m = smf.logit(fml, data=tr).fit(disp=False); pa, pb = m.predict(tr), m.predict(te)
        fpr, tpr, thr = roc_curve(tr["Unhealthy"], pa); cut = thr[np.argmax(tpr - fpr)]
        ya, yb, ha, hb = tr["Unhealthy"].values, te["Unhealthy"].values, (pa >= cut).values, (pb >= cut).values
        return dict(auc=float(roc_auc_score(ya, pa)), cut=float(cut), sens=float((ha & (ya == 1)).sum() / ya.sum()),
                    prec=float((ha & (ya == 1)).sum() / ha.sum()), caught=int((hb & (yb == 1)).sum()), unhealthy=int(yb.sum()),
                    alarms=int((hb & (yb == 0)).sum()))
    P = " + ".join(X_VARS)
    day0 = ns["df"]["Date"].min()
    return dict(n_train=len(tr), n_test=len(te), n_test_all=len(ns["test"]), models=rows,
                nox_zero=int((ns["train"]["NOx"] == 0).sum()),
                logit=dict(base=logi("Unhealthy ~ " + P + " + Winter"), lag=logi("Unhealthy ~ " + P + " + Winter + PM25_lag1")),
                test=dict(day=((te["Date"] - day0).dt.days).tolist(), actual=r(te["PM25"], 1),
                          base=r(preds[specs[0][0]], 1), lag=r(preds[specs[-1][0]], 1)))


if __name__ == "__main__":
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    ns = run_notebook(cells)
    data = collect(ns)
    data["improve"] = improve(ns)
    data["code"] = code_steps(cells)
    data["source"] = NOTEBOOK.name
    block = "/*DATA-START*/" + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "/*DATA-END*/"
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(block, encoding="utf-8")
    else:
        for page in PAGES:
            html = page.read_text(encoding="utf-8")
            html, k = re.subn(r"/\*DATA-START\*/.*?/\*DATA-END\*/", lambda m: block, html, flags=re.S)
            if k != 1:
                sys.exit(f"{page.name} has no data block to refresh")
            page.write_text(html, encoding="utf-8")
    print(f"data block: {len(block) / 1024:.0f} KB")
