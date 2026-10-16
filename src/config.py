from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

TRAIN_CSV = DATA / "train_test.csv"
VAL_CSV = DATA / "validation.csv"
TEMPLATE_CSV = DATA / "validation_predictions_template.csv"
DEC_INPUTS_CSV = DATA / "december_chart_inputs.csv"

# final files (inputs in data/ are never touched)
VAL_PRED_CSV = ROOT / "validation_predictions.csv"
DEC_PRED_CSV = OUT / "december_chart_predictions.csv"

SEED = 42
T0 = "2025-01-01"  # day 0 for the time index

CAT_COLS = ["pickup", "delivery", "equipment"]
# no city ids on purpose: coords-only ties ids+coords on cv, wins the holdout,
# and new cities (12% of val) get handled the same way as known ones
FEATURES = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "equipment", "weight", "quote_signal", "dow",
]

# trend: "none" or "level" (smoothed level in-sample, flat at recent level after).
# level only wins the aug fold (right after the jun peak), loses sep, oct and the
# sep-oct holdout -> off. linear slopes were worse than both
TREND_MODE = "none"
LEVEL_DAYS = 60
SMOOTH_DAYS = 31

# label outliers: rpm off the lane/equip median by more than this factor
OUTLIER_FACTOR = 1.8
OUTLIER_MODE = "keep"  # keep | drop | weight
OUTLIER_WEIGHT = 0.1

HGB_PARAMS = dict(
    loss="absolute_error",
    max_iter=800,
    learning_rate=0.05,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    random_state=SEED,
)

# eval windows: expanding-window folds + the main Sep-Oct holdout
CV_FOLDS = [
    ("2025-08-01", "2025-09-01"),
    ("2025-09-01", "2025-10-01"),
    ("2025-10-01", "2025-11-01"),
]
HOLDOUT = ("2025-09-01", "2025-11-01")
N_HIDDEN_CITIES = 8
N_CITY_REPS = 3
