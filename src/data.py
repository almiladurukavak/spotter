import numpy as np
import pandas as pd

from . import config as C


def load(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def city_coords(*frames):
    # one lat/lon per city, both directions
    parts = []
    for df in frames:
        for side in ["pickup", "delivery"]:
            if f"{side}_lat" in df:
                parts.append(df[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).groupby("city")[["lat", "lon"]].median()


def lane_rpm_resid(df):
    # log rpm minus the lane/equip median -> ~0 for normal rows
    lrpm = np.log(df["posted_rate"] / df["distance"])
    return lrpm - lrpm.groupby([df["pickup"], df["delivery"], df["equipment"]]).transform("median")


def audit(df, name, log):
    """Log data issues w/ counts. Doesn't change anything."""
    def add(issue, n, action):
        log.append({"file": name, "issue": issue, "count": int(n), "action": action})

    add("rows", len(df), "-")
    add("duplicate load_id", df["load_id"].duplicated().sum(), "none found")
    add("duplicate rows (ex load_id)", df.drop(columns="load_id").duplicated().sum(), "none found")
    add("missing weight", df["weight"].isna().sum(), "impute w/ train median per equipment")
    add("negative weight", (df["weight"] < 0).sum(), "abs() - same dist + same rpm as positives, sign glitch")
    add("weight at 47,500 cap", (df["weight"].abs() == 47500).sum(), "keep, looks like clipping")
    add("missing market_index", df["market_index"].isna().sum(), "not used by final model")
    add("distance at 70 mi floor", (df["distance"] == 70).sum(), "keep, consistent per lane")
    ratio = df["distance"] / haversine(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    add("distance > 2x great-circle", (ratio > 2).sum(), "keep, all short lanes on the 70 mi floor")
    add("lat clipped at 25.5", ((df["pickup_lat"] == 25.5) | (df["delivery_lat"] == 25.5)).sum(),
        "keep, distance carries the real info")
    add("lon clipped at -69.5", ((df["pickup_lon"] == -69.5) | (df["delivery_lon"] == -69.5)).sum(),
        "keep, distance carries the real info")
    add("pickup == delivery", (df["pickup"] == df["delivery"]).sum(), "none found")
    coords = city_coords(df)
    add("cities", len(coords), "-")
    if "posted_rate" in df:
        add("non-positive rate", (df["posted_rate"] <= 0).sum(), "none found")
        r = lane_rpm_resid(df)
        hi, lo = (r > np.log(C.OUTLIER_FACTOR)).sum(), (r < -np.log(C.OUTLIER_FACTOR)).sum()
        add(f"rate > {C.OUTLIER_FACTOR}x lane median rpm", hi, f"train: {C.OUTLIER_MODE}")
        add(f"rate < 1/{C.OUTLIER_FACTOR}x lane median rpm", lo, f"train: {C.OUTLIER_MODE}")


def clean(df):
    d = df.copy()
    d["weight"] = d["weight"].abs()
    return d


def outlier_mask(df):
    return (lane_rpm_resid(df).abs() > np.log(C.OUTLIER_FACTOR)).values
