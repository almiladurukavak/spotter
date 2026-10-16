import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C
from .data import outlier_mask
from .features import FeatureBuilder
from .trend import LevelTrend


class LaneMedian:
    """Baseline + fallback: lane/equip median rpm x distance -> global rpm x distance."""

    def fit(self, df):
        rpm = df["posted_rate"] / df["distance"]
        self.lane = rpm.groupby([df["pickup"], df["delivery"], df["equipment"]]).median()
        self.glob = float(rpm.median())
        return self

    def predict(self, df, use_lane=True):
        keys = pd.MultiIndex.from_frame(df[["pickup", "delivery", "equipment"]])
        rpm = self.lane.reindex(keys).to_numpy() if use_lane else np.full(len(df), np.nan)
        src = np.where(np.isnan(rpm), "global_rpm", "lane_rpm")
        rpm = np.where(np.isnan(rpm), self.glob, rpm)
        return rpm * df["distance"].to_numpy(), src


class RateModel:
    """HGB on log(rpm) (minus market level if the trend is on), then x distance."""

    def __init__(self, features=C.FEATURES, trend_mode=C.TREND_MODE, outlier_mode=C.OUTLIER_MODE,
                 hidden_cities=(), params=None):
        self.fb = FeatureBuilder(features, hidden_cities)
        self.trend = LevelTrend(trend_mode)
        self.outlier_mode = outlier_mode
        self.params = {**C.HGB_PARAMS, **(params or {})}

    def fit(self, df):
        df = df.reset_index(drop=True)
        w = np.ones(len(df))
        if self.outlier_mode != "keep":
            bad = outlier_mask(df)
            if self.outlier_mode == "drop":
                df, w = df[~bad].reset_index(drop=True), w[~bad]
            else:
                w[bad] = C.OUTLIER_WEIGHT
        self.fb.fit(df)
        self.trend.fit(df)
        self.fallback = LaneMedian().fit(df)
        y = np.log(df["posted_rate"] / df["distance"]) - self.trend(df["date"])
        self.hgb = HistGradientBoostingRegressor(categorical_features=self.fb.cat_mask, **self.params)
        self.hgb.fit(self.fb.transform(df), y, sample_weight=w)
        return self

    def predict(self, df, return_source=False):
        X = self.fb.transform(df)
        pred = np.exp(self.hgb.predict(X) + self.trend(df["date"])) * df["distance"].to_numpy()
        src = np.full(len(df), "model", dtype=object)
        # fallback chain for anything weird
        bad = ~np.isfinite(pred) | (pred <= 0)
        if bad.any():
            fb_pred, fb_src = self.fallback.predict(df[bad])
            pred[bad], src[bad] = fb_pred, fb_src
        assert np.isfinite(pred).all() and (pred > 0).all()
        return (pred, src) if return_source else pred
