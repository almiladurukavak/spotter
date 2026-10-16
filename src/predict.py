import numpy as np
import pandas as pd

from . import config as C


def predict_validation(model, val):
    tmpl = pd.read_csv(C.TEMPLATE_CSV)
    assert (tmpl["load_id"].to_numpy() == val["load_id"].to_numpy()).all(), "template/val order mismatch"
    pred, src = model.predict(val, return_source=True)
    sub = pd.DataFrame({"load_id": tmpl["load_id"], "predicted_rate": np.round(pred, 2)})
    sub.to_csv(C.VAL_PRED_CSV, index=False)
    return sub, src


def predict_december(model, val):
    raw = pd.read_csv(C.DEC_INPUTS_CSV)
    cols = list(raw.columns)
    d = raw.drop(columns="predicted_rate")
    d["date"] = pd.to_datetime(d["date"])
    d["row"] = np.arange(len(d))
    # no quote_signal/market_index in the chart inputs. val.csv has real values for every
    # dec day -> predict w/ each load's values that day and average. plugging in the daily
    # median was ~5% low on this lane (the model's qs curve has its dip right at the median)
    sig = val[["date", "quote_signal", "market_index"]]
    x = d.merge(sig, on="date", how="left")
    x["p"] = model.predict(x)
    out = raw.copy()
    out["predicted_rate"] = np.round(x.groupby("row")["p"].mean().reindex(d["row"]).to_numpy(), 2)
    out = out[cols]
    C.DEC_PRED_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(C.DEC_PRED_CSV, index=False)
    return out


def check_outputs(sub, dec, train):
    tmpl = pd.read_csv(C.TEMPLATE_CSV)
    p = sub["predicted_rate"]
    assert list(sub.columns) == ["load_id", "predicted_rate"]
    assert len(sub) == 12_000 and (sub["load_id"] == tmpl["load_id"]).all()
    assert p.notna().all() and (p > 0).all()
    assert len(dec) == 31 and dec["predicted_rate"].notna().all() and (dec["predicted_rate"] > 0).all()

    val = pd.read_csv(C.VAL_CSV)
    rpm = p / val["distance"]
    train_rpm = train["posted_rate"] / train["distance"]
    # nothing should be wildly outside what train ever saw
    assert rpm.between(train_rpm.min(), train_rpm.max()).all(), "rpm out of train range"
    return {
        "val_rows": len(sub),
        "val_pred_rate": p.describe().round(2).to_dict(),
        "train_rate": train["posted_rate"].describe().round(2).to_dict(),
        "val_pred_rpm_median": float(rpm.median()),
        "train_rpm_median": float(train_rpm.median()),
        "dec_min": float(dec["predicted_rate"].min()),
        "dec_max": float(dec["predicted_rate"].max()),
    }
