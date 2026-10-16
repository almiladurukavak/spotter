import numpy as np
import pandas as pd

from . import config as C
from .model import LaneMedian, RateModel


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return dict(n=0, MAE=np.nan, RMSE=np.nan, MAPE=np.nan, MedAPE=np.nan)
    e = p - y
    ape = np.abs(e) / y * 100
    return dict(n=len(y), MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), MAPE=ape.mean(), MedAPE=np.median(ape))


def split(df, start, end):
    return df[df["date"] < start], df[(df["date"] >= start) & (df["date"] < end)]


def candidates(variants):
    """name -> fn(train) returning a predict(df) callable."""
    base = {
        "global rpm x dist": lambda a: (lambda b, m=LaneMedian().fit(a): m.predict(b, use_lane=False)[0]),
        "lane median rpm x dist": lambda a: (lambda b, m=LaneMedian().fit(a): m.predict(b)[0]),
    }
    for name, kw in variants.items():
        base[name] = lambda a, kw=kw: RateModel(**kw).fit(a).predict
    return base


def run_windows(df, variants, log=print):
    """Expanding-window folds + holdout. Every metric on ALL rows; clean/seen splits are extra."""
    rows = []
    windows = [("cv", s, e) for s, e in C.CV_FOLDS] + [("holdout", *C.HOLDOUT)]
    for kind, s, e in windows:
        a, b = split(df, s, e)
        seen_lanes = set(zip(a["pickup"], a["delivery"]))
        seen = np.array([k in seen_lanes for k in zip(b["pickup"], b["delivery"])])
        clean = ~b["is_outlier"].to_numpy()
        for name, make in candidates(variants).items():
            p = make(a)(b)
            y = b["posted_rate"].to_numpy()
            for subset, m in [("all", slice(None)), ("clean", clean), ("seen lane", seen), ("unseen lane", ~seen)]:
                rows.append(dict(kind=kind, window=f"{s}..{e}", model=name, subset=subset, **metrics(y[m], p[m])))
            r = rows[-4]
            log(f"  {kind:7s} {s}..{e}  {name:34s} MAE {r['MAE']:7.1f}  RMSE {r['RMSE']:7.1f}  MAPE {r['MAPE']:5.2f}%")
    return pd.DataFrame(rows)


def summarize(res):
    cv = res[res.kind == "cv"].groupby(["model", "subset"])[["MAE", "RMSE", "MAPE", "MedAPE"]].mean().add_prefix("cv_")
    ho = res[res.kind == "holdout"].set_index(["model", "subset"])[["n", "MAE", "RMSE", "MAPE", "MedAPE"]].add_prefix("ho_")
    return cv.join(ho).reset_index()


def city_holdout(df, features, log=print, **kw):
    """Hide N cities from the train vocab, score holdout rows touching them."""
    rng = np.random.RandomState(C.SEED)
    cities = sorted(set(df["pickup"]) | set(df["delivery"]))
    a, b = split(df, *C.HOLDOUT)
    y = b["posted_rate"].to_numpy()
    rows = []
    for rep in range(C.N_CITY_REPS):
        hide = list(rng.choice(cities, C.N_HIDDEN_CITIES, replace=False))
        m = (b["pickup"].isin(hide) | b["delivery"].isin(hide)).to_numpy()
        # lane baseline w/ hidden cities removed from train = global fallback on those rows
        a_vis = a[~(a["pickup"].isin(hide) | a["delivery"].isin(hide))]
        p_base = LaneMedian().fit(a_vis).predict(b)[0]
        p_mod = RateModel(features=features, hidden_cities=hide, **kw).fit(a).predict(b)
        for name, p in [("lane median rpm x dist", p_base), ("model", p_mod)]:
            r = metrics(y[m], p[m])
            rows.append(dict(rep=rep, hidden=", ".join(hide), model=name, **r))
            log(f"  rep{rep} {name:24s} unseen-city rows n={r['n']}  MAE {r['MAE']:6.1f}  MAPE {r['MAPE']:5.2f}%")
    return pd.DataFrame(rows)


def permutation_importance(model, df, n_rep=3):
    """MAE increase on the holdout ($) when a feature is shuffled."""
    rng = np.random.RandomState(C.SEED)
    X = model.fb.transform(df)
    trend = model.trend(df["date"])
    dist, y = df["distance"].to_numpy(), df["posted_rate"].to_numpy()

    def mae(Xp):
        return np.abs(np.exp(model.hgb.predict(Xp) + trend) * dist - y).mean()

    base = mae(X)
    out = []
    for f in X.columns:
        deltas = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            deltas.append(mae(Xp) - base)
        out.append(dict(feature=f, mae_increase=float(np.mean(deltas))))
    return pd.DataFrame(out).sort_values("mae_increase", ascending=False)
