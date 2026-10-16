# spotter

Freight rate prediction take-home for Spotter. Predicts `posted_rate` for the 12,000 validation loads (Nov-Dec 2025) and the fixed December chart for Lexington -> Fort Wayne.

Assessment brief: `docs/freight-rate-ml-assessment-1-.pdf`.

## Setup

Python 3.14 on macOS (should be fine on 3.11+ too). Everything is pip-only, no system deps.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 main.py              # audit -> validation -> retrain on all -> predictions (~2 min)
python3 main.py --fast       # skip validation, just retrain + predict (~25s)
python3 main.py --compare    # also run the model variant grid (~10 min)

python3 score.py --predictions validation_predictions.csv \
                 --december-predictions outputs/december_chart_predictions.csv

python3 make_report.py       # rebuilds report.docx (needs --compare outputs + score.py chart)
```

Outputs:

| file | what |
|---|---|
| `validation_predictions.csv` | final submission, `load_id,predicted_rate`, template order |
| `outputs/december_chart_predictions.csv` | completed december chart inputs (input file in `data/` isn't touched) |
| `scorer_results/candidate_december.png` | chart from `score.py` |
| `outputs/data_quality.csv` | every data check w/ counts |
| `outputs/validation_summary.csv` / `model_comparison.csv` | CV + holdout metrics |
| `outputs/city_holdout.csv` | unseen-city simulation |
| `outputs/feature_importance.csv` | permutation importance on the holdout |
| `outputs/metrics.json` | summary + output sanity checks |

## Structure

```
data/                 provided inputs (unchanged)
docs/                 assessment brief
notebooks/eda.ipynb   EDA
src/
  config.py           paths, features, model + validation settings
  data.py             loading, audit, cleaning
  features.py         feature builder (fit on train only)
  trend.py            market level component (tested, off by default)
  model.py            HGB model + lane-median baseline/fallback
  evaluate.py         time-based CV, holdout, city-holdout sim, importance
  predict.py          writes outputs + sanity checks
  plots.py            shared plots for notebook/report
main.py               entry point
make_report.py        builds report.docx
score.py              provided scorer (unchanged)
```

## Approach

- **Data issues**: negative weights (sign glitch, abs), missing weight (train median per equipment), ~1.4% label outliers at ~0.3x / ~3.5x the lane norm, 70 mi distance floor, clipped coords. All logged in `outputs/data_quality.csv`.
- **Target**: log(rate per mile), multiplied back by distance.
- **Model**: sklearn `HistGradientBoostingRegressor`, absolute-error loss (robust to the noisy labels). Features: pickup/delivery lat/lon, distance, equipment, weight, quote_signal, day of week. No city IDs (coords handle new cities), no month features (Nov/Dec never seen), no market_index (hurt out of time).
- **Trend**: none in the final model. Tested a time-index feature, linear/damped slopes and a market-level component (`src/trend.py`, `TREND_MODE` in config) - the level is a hump not a drift, and none beat no-trend on the Sep-Oct holdout.
- **Fallback**: model -> lane median rpm x distance -> global rpm x distance; asserts no NaN/non-positive output.
- **Validation**: everything forward in time. Holdout = train Jan-Aug, score Sep-Oct; plus 3 expanding-window monthly folds. Encoders/medians/trend fit inside each training window. Then refit on all data.

## Validation results

Sep-Oct holdout = train Jan-Aug, score Sep-Oct (9,523 loads). CV = mean of 3 expanding-window monthly folds (Aug, Sep, Oct). Metrics on **all rows**; last column excludes the ~1.4% label outliers.

| model | CV MAE | CV MAPE | holdout MAE | holdout RMSE | holdout MAPE | holdout MAPE (clean) |
|---|---|---|---|---|---|---|
| **HGB final** | 115.2 | 5.34% | 85.3 | 628.6 | 3.71% | 1.32% |
| HGB, outliers w=0.1 | 115.3 | 5.33% | 85.4 | 628.5 | 3.71% | 1.32% |
| HGB, outliers dropped | 115.4 | 5.35% | 85.7 | 628.6 | 3.72% | 1.33% |
| HGB + city ids | 115.1 | 5.34% | 86.8 | 628.7 | 3.76% | 1.38% |
| HGB + level trend | 112.7 | 5.01% | 90.7 | 629.5 | 3.98% | 1.61% |
| HGB, squared loss | 118.9 | 5.44% | 98.8 | 629.1 | 4.40% | 2.03% |
| HGB + time index | 130.0 | 6.17% | 108.5 | 634.4 | 4.59% | 2.25% |
| HGB + market_index | 136.8 | 6.20% | 127.0 | 639.7 | 5.25% | 2.95% |
| lane median rpm x dist | 153.0 | 6.82% | 152.1 | 673.8 | 6.72% | 4.29% |
| global rpm x dist | 258.4 | 11.57% | 257.0 | 684.2 | 11.63% | 9.39% |

Unseen-city sim (8 random cities hidden from train, 3 repeats): model MAPE 3.89% vs 11.58% for the lane baseline (falls back to global rpm).

RMSE is ~630 for every model - it's almost all the outlier labels. Weak spot: in the Aug fold (right after the Jun peak) the final model was worse than the lane baseline (174 vs 159 MAE); it lags fast level drops.

December chart: $812-825 for Lexington -> Fort Wayne (lane history averaged $823). More detail in `report.docx`.
