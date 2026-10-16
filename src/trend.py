import numpy as np
import pandas as pd

from . import config as C
from .data import lane_rpm_resid
from .features import time_index


class LevelTrend:
    """Market level in log-rpm space.

    In-sample: daily mean of lane-relative residuals (clean rows), smoothed. Out of sample:
    flat at the median of the last LEVEL_DAYS days. Tried linear/damped slopes
    too - the level is a hump (peaks in Jun) so any slope overshot the holdout.
    """

    def __init__(self, mode=C.TREND_MODE, level_days=C.LEVEL_DAYS, smooth_days=C.SMOOTH_DAYS):
        self.mode, self.level_days, self.smooth_days = mode, level_days, smooth_days

    def fit(self, df):
        if self.mode == "none":
            return self
        # lane mean on clean rows, then daily mean of residuals. had median/median first but
        # each lane's middle row is exactly 0 -> daily median got stuck at 0
        lrpm = np.log(df["posted_rate"] / df["distance"])
        ok = (lane_rpm_resid(df).abs() < np.log(C.OUTLIER_FACTOR)).to_numpy()
        keys = [df["pickup"], df["delivery"], df["equipment"]]
        lane_mean = lrpm.where(ok).groupby(keys).transform("mean")
        n = lrpm.where(ok).groupby(keys).transform("count").to_numpy()
        r = (lrpm - lane_mean).to_numpy()
        t = time_index(df["date"])
        # thin lanes give noisy residuals, skip them
        keep = ok & (n >= 3)
        daily = pd.Series(r[keep]).groupby(t[keep]).mean()
        self.tmax = int(t.max())
        full = daily.reindex(np.arange(0, self.tmax + 1))
        self.smooth = full.rolling(self.smooth_days, center=True, min_periods=5).median().ffill().bfill().to_numpy()
        self.level = float(full.iloc[-self.level_days:].median())
        self.daily = daily
        return self

    def __call__(self, dates):
        t = time_index(dates)
        if self.mode == "none":
            return np.zeros(len(t))
        out = np.full(len(t), self.level)
        ins = (t >= 0) & (t <= self.tmax)
        out[ins] = self.smooth[t[ins]]
        return out
