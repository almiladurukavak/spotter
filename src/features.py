import numpy as np
import pandas as pd

from . import config as C
from .data import city_coords


class FeatureBuilder:
    """Everything here is fit on train only and reused for val/december."""

    def __init__(self, features=C.FEATURES, hidden_cities=()):
        self.features = list(features)
        self.hidden = set(hidden_cities)  # for the unseen-city sim

    def _mask_cities(self, df):
        if not self.hidden:
            return df
        df = df.copy()
        for c in ["pickup", "delivery"]:
            df.loc[df[c].isin(self.hidden), c] = "__unseen__"
        return df

    def fit(self, df):
        df = self._mask_cities(df)
        self.cats = {c: sorted(set(df[c]) - {"__unseen__"}) for c in C.CAT_COLS}
        self.weight_med = df.groupby("equipment")["weight"].median()
        self.weight_med_all = df["weight"].median()
        self.qs_med = df["quote_signal"].median()
        self.coords = city_coords(df)
        return self

    def transform(self, df):
        d = df.copy()
        # december inputs come w/o coords -> look them up from train
        for side in ["pickup", "delivery"]:
            for k in ["lat", "lon"]:
                col = f"{side}_{k}"
                looked_up = d[side].map(self.coords[k])
                d[col] = d[col].fillna(looked_up) if col in d else looked_up
        d["weight"] = d["weight"].fillna(d["equipment"].map(self.weight_med)).fillna(self.weight_med_all)
        if "quote_signal" not in d:
            d["quote_signal"] = np.nan
        d["quote_signal"] = d["quote_signal"].fillna(self.qs_med)
        d["dow"] = d["date"].dt.dayofweek
        d["t"] = time_index(d["date"])  # only used by the time-index variant
        d = self._mask_cities(d)
        X = d[self.features].copy()
        for c in C.CAT_COLS:
            if c in X:
                # unseen city -> NaN, HGB treats it as missing
                codes = pd.Categorical(X[c], categories=self.cats[c]).codes.astype(float)
                X[c] = np.where(codes < 0, np.nan, codes)
        return X

    @property
    def cat_mask(self):
        return [f in C.CAT_COLS for f in self.features]


def time_index(dates):
    return (pd.to_datetime(dates) - pd.Timestamp(C.T0)).dt.days.to_numpy()
