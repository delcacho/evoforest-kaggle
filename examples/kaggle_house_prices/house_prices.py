"""Kaggle "House Prices - Advanced Regression Techniques" on top of EvoForest.

Three subcommands:

``prepare``
    Turn Kaggle's ``train.csv`` / ``test.csv`` into a dense, standardized numeric
    matrix, write it as a row-aligned ``.npz`` plus an ``external-manifest`` JSON that
    ``evoforest-arch evolve`` can consume, and write a hand-authored task-context brief
    (column index -> feature name) so the LLM scientist/engineer can reason about
    columns by meaning rather than by anonymous index.

``baseline``
    No-LLM path: build the generic tabular seed graph, let ``RidgeEvaluator`` pick the
    best configuration by K-fold RMSE on log(SalePrice), then refit on every training
    row and write ``submission.csv``. Useful as a sanity floor before spending LLM budget.

``predict``
    Load the best graph and configuration from a finished ``evoforest-arch evolve`` run,
    refit the Ridge readout on *all* Kaggle training rows, and write ``submission.csv``.

The competition metric is RMSE between log(predicted) and log(actual) price, so the
regression target is ``log1p(SalePrice)`` and every reported score is directly
comparable to the leaderboard.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.graph import Graph
from evoforest_arch.graph_io import graph_from_path
from evoforest_arch.primitives import PrimitiveRegistry
from evoforest_arch.readout import Standardizer
from evoforest_arch.seed import build_tabular_seed_graph
from evoforest_arch.task import TaskSchema

INPUT_NAME = "x"
TARGET = "SalePrice"
ID_COLUMN = "Id"

TRAIN_NPZ = "house_prices_train.npz"
TEST_NPZ = "house_prices_test.npz"
MANIFEST = "dataset_manifest.json"
TASK_CONTEXT = "task_context.md"
FEATURES_JSON = "feature_names.json"

# ---------------------------------------------------------------------------
# Domain knowledge from the competition's data_description.txt
# ---------------------------------------------------------------------------

_QUALITY = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPS: dict[str, dict[str, int]] = {
    **{
        column: _QUALITY
        for column in (
            "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
            "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
        )
    },
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
}

# NA means "feature absent" for these columns, not "unknown".
NONE_IF_MISSING = (
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
)
ZERO_IF_MISSING = (
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
)
# Numeric codes that are really categories.
NUMERIC_AS_CATEGORICAL = ("MSSubClass", "MoSold")
# Unseen rows below the training minimum are clipped here so features stay strictly positive.
POSITIVE_FLOOR = 0.1


@dataclass(frozen=True)
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    test_ids: np.ndarray
    feature_names: list[str]


def _col(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(default, index=frame.index, dtype=np.float64)


def _clean(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using statistics from ``reference`` (the training rows only)."""

    frame = frame.copy()
    for column in NONE_IF_MISSING:
        if column in frame:
            frame[column] = frame[column].fillna("None")
    for column in ZERO_IF_MISSING:
        if column in frame:
            frame[column] = frame[column].fillna(0.0)
    if "GarageYrBlt" in frame:
        frame["GarageYrBlt"] = frame["GarageYrBlt"].fillna(_col(frame, "YearBuilt"))
    if "LotFrontage" in frame:
        overall = float(reference["LotFrontage"].median()) if reference["LotFrontage"].notna().any() else 0.0
        if "Neighborhood" in frame:
            by_hood = reference.groupby("Neighborhood")["LotFrontage"].median()
            frame["LotFrontage"] = frame["LotFrontage"].fillna(frame["Neighborhood"].map(by_hood))
        frame["LotFrontage"] = frame["LotFrontage"].fillna(overall)
    for column in NUMERIC_AS_CATEGORICAL:
        if column in frame:
            frame[column] = frame[column].astype("Int64").astype(str)
    for column in frame.columns:
        if frame[column].isna().any():
            ref = reference[column] if column in reference else frame[column]
            if frame[column].dtype == object:
                mode = ref.mode(dropna=True)
                frame[column] = frame[column].fillna(mode.iloc[0] if len(mode) else "None")
            else:
                median = ref.median()
                frame[column] = frame[column].fillna(0.0 if pd.isna(median) else median)
    for column, mapping in ORDINAL_MAPS.items():
        if column in frame:
            frame[column] = frame[column].map(mapping).fillna(0).astype(np.float64)
    return frame


def _engineer(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["TotalSF"] = _col(frame, "TotalBsmtSF") + _col(frame, "1stFlrSF") + _col(frame, "2ndFlrSF")
    frame["TotalBath"] = (
        _col(frame, "FullBath") + 0.5 * _col(frame, "HalfBath")
        + _col(frame, "BsmtFullBath") + 0.5 * _col(frame, "BsmtHalfBath")
    )
    frame["TotalPorchSF"] = (
        _col(frame, "OpenPorchSF") + _col(frame, "EnclosedPorch") + _col(frame, "3SsnPorch")
        + _col(frame, "ScreenPorch") + _col(frame, "WoodDeckSF")
    )
    if "YrSold" in frame:
        frame["HouseAge"] = (frame["YrSold"] - _col(frame, "YearBuilt")).clip(lower=0)
        frame["RemodAge"] = (frame["YrSold"] - _col(frame, "YearRemodAdd")).clip(lower=0)
    frame["HasGarage"] = (_col(frame, "GarageArea") > 0).astype(np.float64)
    frame["HasBsmt"] = (_col(frame, "TotalBsmtSF") > 0).astype(np.float64)
    frame["Has2ndFlr"] = (_col(frame, "2ndFlrSF") > 0).astype(np.float64)
    frame["HasPool"] = (_col(frame, "PoolArea") > 0).astype(np.float64)
    frame["HasFireplace"] = (_col(frame, "Fireplaces") > 0).astype(np.float64)
    return frame


def prepare_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    drop_outliers: bool = True,
    min_category_count: int = 10,
    skew_threshold: float = 0.75,
) -> PreparedData:
    """Leakage-safe preprocessing: every statistic is computed on training rows only."""

    if TARGET not in train:
        raise ValueError(f"train frame must contain {TARGET!r}.")
    train = train.copy()
    if drop_outliers and {"GrLivArea", TARGET} <= set(train.columns):
        # The dataset author (De Cock, 2011) recommends removing these partial sales.
        train = train[~((train["GrLivArea"] > 4000) & (train[TARGET] < 300_000))]
    y = np.log1p(train[TARGET].to_numpy(dtype=np.float64))
    test_ids = test[ID_COLUMN].to_numpy() if ID_COLUMN in test else np.arange(len(test))
    raw_train = train.drop(columns=[TARGET, ID_COLUMN], errors="ignore").reset_index(drop=True)
    raw_test = test.drop(columns=[TARGET, ID_COLUMN], errors="ignore").reset_index(drop=True)

    tr = _engineer(_clean(raw_train, raw_train))
    te = _engineer(_clean(raw_test, raw_train))

    categorical = [c for c in tr.columns if tr[c].dtype == object]
    numeric = [c for c in tr.columns if c not in categorical]
    ordinal = set(ORDINAL_MAPS)

    # Log-transform right-skewed, non-negative, non-ordinal numeric columns.
    for column in numeric:
        if column in ordinal or tr[column].min() < 0 or tr[column].nunique() <= 2:
            continue
        if abs(float(tr[column].skew())) > skew_threshold:
            tr[column] = np.log1p(tr[column])
            te[column] = np.log1p(te[column].clip(lower=0))

    # One-hot encode categories frequent enough in train; rare/unseen levels map to all-zero.
    blocks_tr = [tr[numeric].astype(np.float64)]
    blocks_te = [te[numeric].astype(np.float64)]
    for column in categorical:
        counts = tr[column].value_counts()
        levels = sorted(counts[counts >= min_category_count].index)
        if len(levels) < 2:
            continue
        names = [f"{column}={level}" for level in levels]
        blocks_tr.append(pd.DataFrame({n: (tr[column] == lv).astype(np.float64) for n, lv in zip(names, levels)}))
        blocks_te.append(pd.DataFrame({n: (te[column] == lv).astype(np.float64) for n, lv in zip(names, levels)}))
    x_tr = pd.concat(blocks_tr, axis=1)
    x_te = pd.concat(blocks_te, axis=1).reindex(columns=x_tr.columns, fill_value=0.0)

    # Positive unit-spread scaling with train statistics: column = 1 + (value - train_min) / train_std.
    # Every training value is >= 1, so LLM-written log() / ratio features are well defined
    # (z-scores sit around 0, which made log1p() and divisions blow up on unseen rows).
    # Ridge standardizes features internally, so the linear part of the model is unaffected.
    low = x_tr.min(axis=0)
    std = x_tr.std(axis=0, ddof=0)
    keep = std > 1e-12
    x_tr = 1.0 + (x_tr.loc[:, keep] - low[keep]) / std[keep]
    x_te = (1.0 + (x_te.loc[:, keep] - low[keep]) / std[keep]).clip(lower=POSITIVE_FLOOR)

    # Order columns by |corr(feature, target)| so the seed graph's low-rank interaction
    # primitives (which only look at the first few columns) pair the strongest signals.
    corr = x_tr.apply(lambda col: abs(np.corrcoef(col.to_numpy(), y)[0, 1]) if col.std() > 0 else 0.0)
    order = corr.fillna(0.0).sort_values(ascending=False, kind="stable").index
    x_tr = x_tr[order]
    x_te = x_te[order]

    x_train = np.nan_to_num(x_tr.to_numpy(dtype=np.float64))
    x_test = np.nan_to_num(x_te.to_numpy(dtype=np.float64))
    return PreparedData(x_train, y, x_test, test_ids, [str(c) for c in x_tr.columns])


def task_schema() -> TaskSchema:
    return TaskSchema.tabular(input_name=INPUT_NAME, name="kaggle-house-prices")


def write_prepared(data: PreparedData, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / TRAIN_NPZ, **{INPUT_NAME: data.x_train, "y": data.y_train})
    np.savez(out_dir / TEST_NPZ, **{INPUT_NAME: data.x_test, "Id": data.test_ids})
    (out_dir / FEATURES_JSON).write_text(json.dumps(data.feature_names, indent=1), encoding="utf-8")
    manifest = {
        "adapter": "external-npz",
        "path": TRAIN_NPZ,
        "target_key": "y",
        "input_keys": [INPUT_NAME],
        "dataset_name": "kaggle-house-prices",
        "task_schema": task_schema().to_dict(),
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / TASK_CONTEXT).write_text(task_context_text(data), encoding="utf-8")


def task_context_text(data: PreparedData) -> str:
    lines = [
        "# Task: Kaggle House Prices (Ames, Iowa residential sales)",
        "",
        f"Target `y` is log1p(SalePrice) for {len(data.y_train)} sold homes; the competition metric is",
        "RMSE on this log scale, so lower RMSE is better and multiplicative price effects are additive here.",
        "",
        f"Input `{INPUT_NAME}` is a ({data.x_train.shape[0]}, {data.x_train.shape[1]}) float matrix. Every column is scaled as",
        "`1 + (value - train_min) / train_std`: the smallest training value is exactly 1, one unit equals one",
        f"training standard deviation, larger = more, and unseen rows are clipped at {POSITIVE_FLOOR}. Columns are strictly",
        "positive, so np.log(v), np.sqrt(v) and ratios a / b are well defined; still prefer bounded forms",
        "(np.log(v), a / (a + b)) over raw ratios, which amplify small denominators.",
        "Right-skewed areas/counts were log1p-transformed before scaling. Quality/condition grades are ordinal",
        "(higher = better). `Name=Level` one-hot indicators take values 1 (absent) or 1 + 1/std (present).",
        "Columns are sorted by |correlation with y|, strongest first.",
        "",
        "Domain hints: price scales with quality x living area; neighborhood shifts the price level;",
        "age/remodel recency and garage/basement presence matter; very large lots and areas saturate.",
        "",
        "## Column profile (training rows, after scaling)",
        "",
        "kind: bin = two values (indicator), disc = <= 15 distinct values (grades, counts), cont = continuous.",
        "at_min = share of rows at the column minimum 1.0; for areas/counts that means zero/absent",
        "(zero-inflated columns: a ratio or log carries little signal for those rows).",
        "corr = Pearson correlation with y (sign tells direction). skew > 1 = long right tail.",
        f"hist: continuous columns use {HIST_BINS} equal-width bins from min to p99 (rows above p99 in the last bin);",
        "discrete columns use one digit per observed level in increasing order. Each digit is the count relative",
        "to the tallest bin (0 = empty or < 5%, 9 = tallest). The rows at the minimum are",
        "excluded from hist when at_min >= 20%, so it shows the shape of the non-zero part.",
        "shape = uni / bi / multi(modal) from peaks of the smoothed histogram, '-' for indicators.",
        "",
        "| idx | feature | kind | distinct | at_min | median | p99 | max | skew | corr | hist | shape |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += column_profile_rows(data.x_train, data.y_train, data.feature_names)
    return "\n".join(lines) + "\n"


def column_profile_rows(x: np.ndarray, y: np.ndarray, names: list[str]) -> list[str]:
    """One markdown row per column with train-only distribution statistics."""
    rows = []
    for i, name in enumerate(names):
        col = x[:, i]
        distinct = int(np.unique(col).size)
        kind = "bin" if distinct <= 2 else "disc" if distinct <= 15 else "cont"
        at_min = float(np.mean(np.isclose(col, col.min())))
        sd = col.std()
        skew = float(np.mean(((col - col.mean()) / sd) ** 3)) if sd > 0 else 0.0
        corr = float(np.corrcoef(col, y)[0, 1]) if sd > 0 else 0.0
        hist, shape = histogram_profile(col)
        rows.append(
            f"| {i} | {name} | {kind} | {distinct} | {at_min:.1%} | {np.median(col):.2f} | "
            f"{np.quantile(col, 0.99):.2f} | {col.max():.2f} | {skew:+.1f} | {corr:+.2f} | "
            f"{hist} | {shape} |"
        )
    return rows


HIST_BINS = 12


def histogram_profile(col: np.ndarray) -> tuple[str, str]:
    """Digit histogram (0-9 relative to the tallest bin) and a uni/bi/multi modality label."""
    values = col
    if np.mean(np.isclose(col, col.min())) >= 0.2:
        values = col[~np.isclose(col, col.min())]
    levels, level_counts = np.unique(values, return_counts=True)
    if levels.size <= 2 or values.size < 30:
        return "-", "-"
    if levels.size <= 15:
        # Discrete grades/counts: one digit per observed level, so gaps between levels aren't read as valleys.
        counts = level_counts
    else:
        low, high = float(values.min()), float(np.quantile(values, 0.99))
        if high <= low:
            high = float(values.max())
        counts, _ = np.histogram(np.clip(values, low, high), bins=HIST_BINS, range=(low, high))
    digits = "".join(
        "0" if c < 0.05 * counts.max() else str(min(9, int(round(9 * c / counts.max())))) for c in counts
    )
    smooth = np.convolve(np.pad(counts.astype(float), 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    peaks = [
        i for i in range(len(smooth))
        if smooth[i] >= 0.15 * smooth.max()
        and (i == 0 or smooth[i] > smooth[i - 1])
        and (i == len(smooth) - 1 or smooth[i] >= smooth[i + 1])
    ]
    # Two neighbouring peaks only count separately if the valley between them is a real dip.
    modes = 1 if peaks else 0
    for a, b in zip(peaks, peaks[1:]):
        if smooth[a:b + 1].min() < 0.7 * min(smooth[a], smooth[b]):
            modes += 1
    shape = "uni" if modes <= 1 else "bi" if modes == 2 else "multi"
    return digits, shape


def load_prepared(data_dir: pathlib.Path) -> PreparedData:
    with np.load(data_dir / TRAIN_NPZ, allow_pickle=False) as train:
        x_train, y_train = train[INPUT_NAME], train["y"]
    with np.load(data_dir / TEST_NPZ, allow_pickle=False) as test:
        x_test, ids = test[INPUT_NAME], test["Id"]
    names = json.loads((data_dir / FEATURES_JSON).read_text(encoding="utf-8"))
    return PreparedData(x_train, y_train, x_test, ids, names)


# ---------------------------------------------------------------------------
# Fitting and prediction
# ---------------------------------------------------------------------------


def fit_predict(
    graph: Graph,
    config: dict[str, str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    evaluator: RidgeEvaluator | None = None,
) -> np.ndarray:
    """Refit the Ridge readout on all training rows and predict the test rows.

    Several tabular primitives (and LLM-written lambdas) use batch statistics such as
    column means. To give train and test rows the exact same feature transform, the graph
    is evaluated once on the stacked train+test matrix (unsupervised use of test features
    only; the target is never touched). Sample/residual-weight rules are evaluated on the
    training rows alone because they size their output to the rows being fitted.
    """

    evaluator = evaluator or RidgeEvaluator(refine_globals=False, diagnostics_mode="basic")
    frozen = graph.clone()
    selected = frozen.selected_config(config)
    n_train = x_train.shape[0]

    stacked = np.vstack([x_train, x_test])
    features, _, _ = frozen.evaluate_features({INPUT_NAME: stacked}, config=selected)
    _, _, train_ctx = frozen.evaluate_features({INPUT_NAME: x_train}, config=selected)
    sample_weight, residual_rule, _ = evaluator._evaluate_fitting_rules(
        frozen, {INPUT_NAME: x_train}, y_train, selected, train_ctx
    )
    standardizer = Standardizer.fit(features[:n_train])
    fit = evaluator._fit_reweighted_ridge(
        standardizer.transform(features[:n_train]), y_train, sample_weight, residual_rule
    )
    return fit.model.predict(standardizer.transform(features[n_train:]))


def write_submission(path: pathlib.Path, ids: np.ndarray, log_predictions: np.ndarray) -> None:
    prices = np.expm1(np.asarray(log_predictions, dtype=np.float64))
    if not np.isfinite(prices).all():
        raise ValueError("Non-finite predictions; refusing to write a submission.")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COLUMN: ids.astype(int), TARGET: prices}).to_csv(path, index=False)


def make_evaluator(folds: int, seed: int, refine_globals: bool) -> RidgeEvaluator:
    return RidgeEvaluator(
        n_splits=folds,
        seed=seed,
        scorer="rmse",
        refine_globals=refine_globals,
        refine_backend="auto",
        diagnostics_mode="basic",
        task_schema=task_schema(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> int:
    train = pd.read_csv(args.kaggle_dir / "train.csv")
    test = pd.read_csv(args.kaggle_dir / "test.csv")
    data = prepare_frames(
        train, test, drop_outliers=not args.keep_outliers, min_category_count=args.min_category_count
    )
    write_prepared(data, args.out)
    print(f"train: {data.x_train.shape}  test: {data.x_test.shape}  -> {args.out}")
    print(f"top features: {', '.join(data.feature_names[:6])}")
    print(f"manifest: {args.out / MANIFEST}\ntask context: {args.out / TASK_CONTEXT}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    data = load_prepared(args.data)
    graph = build_tabular_seed_graph(task_schema())
    evaluator = make_evaluator(args.folds, args.seed, args.refine_globals)
    result = evaluator.evaluate(graph, {INPUT_NAME: data.x_train}, data.y_train, update_graph=True)
    raw_rmse = -float(result.score)
    print(f"seed graph {args.folds}-fold CV RMSE(log price): {raw_rmse:.5f}")
    print(f"selected config: {json.dumps(result.config, sort_keys=True)}")
    preds = fit_predict(graph, result.config, data.x_train, data.y_train, data.x_test, evaluator=evaluator)
    write_submission(args.submission, data.test_ids, preds)
    print(f"wrote {args.submission} ({len(preds)} rows)")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    data = load_prepared(args.data)
    state = json.loads((args.run_dir / "state.json").read_text(encoding="utf-8"))
    graph_path = args.run_dir / state.get("best_graph_path", "best_graph.json")
    registry = PrimitiveRegistry.for_task(task_schema())
    graph = graph_from_path(graph_path, registry=registry, allow_source=args.allow_source)
    config = {str(k): str(v) for k, v in dict(state.get("best_config", {})).items()}
    evaluator = make_evaluator(args.folds, args.seed, refine_globals=False)
    if args.cv:
        # Re-score on the full Kaggle train set (includes the run's validation/test rows).
        result = evaluator.evaluate(graph, {INPUT_NAME: data.x_train}, data.y_train, config=config)
        print(f"best graph {args.folds}-fold CV RMSE(log price) on all train rows: {-float(result.score):.5f}")
    preds = fit_predict(graph, config, data.x_train, data.y_train, data.x_test, evaluator=evaluator)
    write_submission(args.submission, data.test_ids, preds)
    print(f"graph: {graph_path}\nwrote {args.submission} ({len(preds)} rows)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    here = pathlib.Path(__file__).resolve().parent
    default_data = here / "data" / "prepared"
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="CSV -> standardized npz + evolve manifest")
    prep.add_argument("--kaggle-dir", type=pathlib.Path, default=here / "data" / "raw")
    prep.add_argument("--out", type=pathlib.Path, default=default_data)
    prep.add_argument("--keep-outliers", action="store_true")
    prep.add_argument("--min-category-count", type=int, default=10)
    prep.set_defaults(func=cmd_prepare)

    base = sub.add_parser("baseline", help="seed graph, no LLM, write submission")
    base.add_argument("--data", type=pathlib.Path, default=default_data)
    base.add_argument("--folds", type=int, default=10)
    base.add_argument("--seed", type=int, default=0)
    base.add_argument("--refine-globals", action="store_true", help="Try PyTorch L-BFGS global refinement.")
    base.add_argument("--submission", type=pathlib.Path, default=here / "submissions" / "baseline.csv")
    base.set_defaults(func=cmd_baseline)

    pred = sub.add_parser("predict", help="best graph from an evolve run -> submission")
    pred.add_argument("run_dir", type=pathlib.Path)
    pred.add_argument("--data", type=pathlib.Path, default=default_data)
    pred.add_argument("--folds", type=int, default=10)
    pred.add_argument("--seed", type=int, default=0)
    pred.add_argument("--cv", action="store_true", help="Also report CV RMSE of the evolved graph on all train rows.")
    pred.add_argument("--allow-source", action="store_true", help="Required if the graph holds LLM-written lambdas.")
    pred.add_argument("--submission", type=pathlib.Path, default=here / "submissions" / "evolved.csv")
    pred.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
