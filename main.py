"""End to end: audit -> validate -> retrain on all -> predict -> sanity checks.

python3 main.py            # full run
python3 main.py --fast     # skip validation, just retrain + predict
python3 main.py --compare  # also run the model variant grid (slow)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config as C
from src.data import audit, clean, load, outlier_mask
from src.evaluate import city_holdout, permutation_importance, run_windows, split, summarize
from src.model import RateModel
from src.predict import check_outputs, predict_december, predict_validation

VARIANTS = {
    # name -> RateModel kwargs; first one is the final config
    "HGB final": dict(),
}
COMPARE = {
    "HGB + time index": dict(features=C.FEATURES + ["t"]),
    "HGB + level trend": dict(trend_mode="level"),
    "HGB + market_index": dict(features=C.FEATURES + ["market_index"]),
    "HGB + city ids": dict(features=["pickup", "delivery"] + C.FEATURES),
    "HGB, outliers dropped": dict(outlier_mode="drop"),
    "HGB, outliers w=0.1": dict(outlier_mode="weight"),
    "HGB, squared loss": dict(params=dict(loss="squared_error")),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    C.OUT.mkdir(exist_ok=True)

    train, val = load(C.TRAIN_CSV), load(C.VAL_CSV)
    log = []
    audit(train, "train_test.csv", log)
    audit(val, "validation.csv", log)
    dq = pd.DataFrame(log)
    dq.to_csv(C.OUT / "data_quality.csv", index=False)
    print("== data quality")
    print(dq.to_string(index=False))

    train, val = clean(train), clean(val)
    train["is_outlier"] = outlier_mask(train)
    summary = {}

    if not args.fast:
        variants = {**VARIANTS, **(COMPARE if args.compare else {})}
        print("\n== validation (expanding window + Sep-Oct holdout)")
        res = run_windows(train, variants)
        res.to_csv(C.OUT / "validation_windows.csv", index=False)
        tab = summarize(res)
        tab[tab.model.isin(["HGB final", "lane median rpm x dist", "global rpm x dist"])].to_csv(
            C.OUT / "validation_summary.csv", index=False)
        if args.compare:
            tab.to_csv(C.OUT / "model_comparison.csv", index=False)
        print(tab[tab.subset == "all"].round(2).to_string(index=False))

        print("\n== unseen-city sim")
        ch = city_holdout(train, C.FEATURES)
        ch.to_csv(C.OUT / "city_holdout.csv", index=False)

        print("\n== feature importance (holdout model)")
        a, b = split(train, *C.HOLDOUT)
        fi = permutation_importance(RateModel().fit(a), b)
        fi.to_csv(C.OUT / "feature_importance.csv", index=False)
        print(fi.round(1).to_string(index=False))

        final = tab[(tab.model == "HGB final")].set_index("subset")
        summary["validation"] = final[["cv_MAE", "cv_MAPE", "ho_MAE", "ho_RMSE", "ho_MAPE"]].round(3).to_dict("index")
        summary["city_holdout"] = ch.groupby("model")[["MAE", "MAPE"]].mean().round(3).to_dict("index")

    print("\n== final fit on all train data")
    model = RateModel().fit(train)
    sub, src = predict_validation(model, val)
    dec = predict_december(model, val)
    summary["trend_mode"] = C.TREND_MODE
    summary["prediction_sources"] = pd.Series(src).value_counts().to_dict()
    summary["checks"] = check_outputs(sub, dec, train)
    (C.OUT / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary["checks"], indent=2, default=float))
    print(f"\ndone in {time.time() - t0:.0f}s -> {C.VAL_PRED_CSV.name}, {C.DEC_PRED_CSV.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
