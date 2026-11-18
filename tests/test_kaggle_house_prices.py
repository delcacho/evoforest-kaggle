from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from evoforest_arch.datasets import load_dataset_bundle
from evoforest_arch.seed import build_tabular_seed_graph

_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "kaggle_house_prices" / "house_prices.py"
_spec = importlib.util.spec_from_file_location("house_prices_example", _PATH)
hp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = hp
_spec.loader.exec_module(hp)


def _fake_frames(n_train: int = 120, n_test: int = 30, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    qual = rng.integers(1, 11, n)
    area = rng.lognormal(7.2, 0.3, n)
    frame = pd.DataFrame(
        {
            "Id": np.arange(1, n + 1),
            "MSSubClass": rng.choice([20, 60, 120], n),
            "Neighborhood": rng.choice(["A", "B", "C"], n),
            "LotFrontage": np.where(rng.random(n) < 0.2, np.nan, rng.normal(70, 10, n)),
            "OverallQual": qual,
            "GrLivArea": area,
            "TotalBsmtSF": rng.uniform(0, 1500, n),
            "1stFlrSF": area * 0.6,
            "2ndFlrSF": area * 0.4,
            "ExterQual": rng.choice(["TA", "Gd", "Ex"], n),
            "PoolQC": np.where(rng.random(n) < 0.95, None, "Gd"),
            "GarageArea": rng.uniform(0, 800, n),
            "YearBuilt": rng.integers(1900, 2008, n),
            "YrSold": 2009,
        }
    )
    frame["SalePrice"] = np.exp(10.5 + 0.1 * qual + 0.5 * np.log(area) + rng.normal(0, 0.05, n))
    train = frame.iloc[:n_train].reset_index(drop=True)
    test = frame.iloc[n_train:].drop(columns=["SalePrice"]).reset_index(drop=True)
    return train, test


def test_prepare_is_finite_aligned_and_train_standardized() -> None:
    train, test = _fake_frames()
    data = hp.prepare_frames(train, test)
    assert data.x_train.shape[1] == data.x_test.shape[1] == len(data.feature_names)
    assert data.x_train.shape[0] == len(data.y_train) == len(train)
    assert np.isfinite(data.x_train).all() and np.isfinite(data.x_test).all()
    np.testing.assert_allclose(data.x_train.min(axis=0), 1.0, atol=1e-9)  # positive scaling: train min is 1
    np.testing.assert_allclose(data.x_train.std(axis=0), 1.0, atol=1e-9)
    assert (data.x_test >= hp.POSITIVE_FLOOR).all()
    assert "PoolQC" in data.feature_names  # ordinal-encoded, NA treated as "None"
    assert any(name.startswith("Neighborhood=") for name in data.feature_names)
    np.testing.assert_allclose(data.y_train, np.log1p(train["SalePrice"]))


def test_manifest_loads_and_seed_graph_predicts_submission(tmp_path: pathlib.Path) -> None:
    train, test = _fake_frames()
    data = hp.prepare_frames(train, test)
    hp.write_prepared(data, tmp_path)

    bundle = load_dataset_bundle({"name": "external-manifest", "manifest_path": str(tmp_path / hp.MANIFEST)})
    assert bundle.task_schema.kind == "tabular"
    assert bundle.inputs[hp.INPUT_NAME].shape == data.x_train.shape

    graph = build_tabular_seed_graph(hp.task_schema())
    evaluator = hp.make_evaluator(folds=3, seed=0, refine_globals=False)
    result = evaluator.evaluate(graph, {hp.INPUT_NAME: data.x_train}, data.y_train)
    assert -result.score < np.std(data.y_train)  # beats the constant predictor

    preds = hp.fit_predict(graph, result.config, data.x_train, data.y_train, data.x_test, evaluator=evaluator)
    out = tmp_path / "submission.csv"
    hp.write_submission(out, data.test_ids, preds)
    submission = pd.read_csv(out)
    assert list(submission.columns) == ["Id", "SalePrice"]
    assert submission["Id"].tolist() == test["Id"].tolist()
    assert (submission["SalePrice"] > 0).all()
    assert json.loads((tmp_path / hp.FEATURES_JSON).read_text()) == data.feature_names
