from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from evoforest_arch.production import (
    CV_SIGN_TEST_PROMOTION_POLICY,
    ProductionConfig,
    ProductionEvolutionRunner,
    fold_sign_test,
)


def test_sign_test_thresholds_for_ten_folds() -> None:
    best = np.zeros(10)
    # One-sided P(X >= k | n=10, p=1/2): 9 -> 0.0107, 8 -> 0.0547, 7 -> 0.1719.
    assert fold_sign_test([1.0] * 9 + [-1.0], best, alpha=0.05)["accept"]
    assert not fold_sign_test([1.0] * 8 + [-1.0] * 2, best, alpha=0.05)["accept"]
    assert fold_sign_test([1.0] * 8 + [-1.0] * 2, best, alpha=0.10)["accept"]
    seven = fold_sign_test([1.0] * 7 + [-1.0] * 3, best, alpha=0.05)
    assert not seven["accept"]
    assert seven["p_value"] == pytest.approx(176 / 1024)


def test_sign_test_drops_ties_and_rejects_mismatched_folds() -> None:
    result = fold_sign_test([1.0] * 5 + [0.0] * 5, np.zeros(10), alpha=0.05)
    assert result["ties"] == 5 and result["wins"] == 5
    assert result["accept"]  # 5/5 non-tied wins: p = 1/32
    assert not fold_sign_test([1.0] * 5, np.zeros(10), alpha=0.05)["accept"]
    assert not fold_sign_test(np.zeros(10), np.zeros(10), alpha=0.05)["accept"]


def _result(fold_scores: list[float]) -> SimpleNamespace:
    return SimpleNamespace(score=float(np.mean(fold_scores)), diagnostics={"folds": {"score": fold_scores}})


def _promotes(tmp_path: pathlib.Path, candidate_folds: list[float], candidate_validation: float) -> bool:
    config = ProductionConfig(
        output_dir=tmp_path,
        promotion_policy=CV_SIGN_TEST_PROMOTION_POLICY,
        sign_test_alpha=0.05,
        max_validation_regression=0.10,
    )
    runner = SimpleNamespace(config=config, _acceptance_policy_name=lambda: "cv_fold_sign_test")
    runner._record_promotion_decision = ProductionEvolutionRunner._record_promotion_decision
    context = SimpleNamespace(
        run_dir=tmp_path,
        manifest={"acceptance": {"policy": "cv_fold_sign_test", "sign_test_alpha": 0.05, "max_validation_regression": 0.10}},
        best_train_result=_result([-0.12] * 10),
        best_validation_result=SimpleNamespace(score=-0.14),
    )
    return ProductionEvolutionRunner._promotes(
        runner, _result(candidate_folds), SimpleNamespace(score=candidate_validation), context
    )


def test_sign_test_policy_accepts_consistent_gain_and_logs_decision(tmp_path: pathlib.Path) -> None:
    assert _promotes(tmp_path, [-0.11] * 9 + [-0.13], candidate_validation=-0.141)
    row = json.loads((tmp_path / "promotion_decisions.jsonl").read_text().splitlines()[-1])
    assert row["accepted"] and row["sign_test"]["wins"] == 9


def test_sign_test_policy_rejects_lucky_average(tmp_path: pathlib.Path) -> None:
    # Mean improves (one huge fold) but only 2/10 folds win.
    assert not _promotes(tmp_path, [-0.02, -0.05] + [-0.121] * 8, candidate_validation=-0.13)


def test_validation_regression_veto_blocks_blowups(tmp_path: pathlib.Path) -> None:
    # Every fold improves, but validation is 6x worse (like a log/ratio feature exploding).
    assert not _promotes(tmp_path, [-0.11] * 10, candidate_validation=-0.90)
    row = json.loads((tmp_path / "promotion_decisions.jsonl").read_text().splitlines()[-1])
    assert row["validation_vetoed"] and row["sign_test"]["accept"]


def test_llm_engine_treats_declared_column_count_as_advisory() -> None:
    from evoforest_arch.mutations import MutationDocument, MutationEngine, MutationSpec
    from evoforest_arch.seed import build_structural_break_seed_graph
    from evoforest_arch.synthetic import make_structural_break_data

    dataset = make_structural_break_data(n_series=24, length=70, seed=34)
    graph = build_structural_break_seed_graph()
    document = MutationDocument(
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="output",
                primitive="source",
                alternative_id="whole_block_log",
                parents=("segment_stats",),
                source="lambda ctx, values: np.log1p(np.abs(values['segment_stats'].values))",
                output_contract={"type": "feature_block", "n_columns": 1},  # wrong, as LLMs often write
            ),
        )
    )
    applied = MutationEngine(allow_source=True, advisory_column_counts=True).apply_document(graph, document).graph
    applied.evaluate_features(dataset.inputs(), applied.default_config())  # strict mode raises here


def test_standardizer_clips_extrapolation_from_near_constant_columns() -> None:
    from evoforest_arch.readout import Standardizer

    train = np.column_stack([np.linspace(0, 1, 50), np.full(50, 1.0) + np.r_[np.zeros(49), 1e-6]])
    held_out = np.array([[0.5, 1.0 + 1e-2]])  # tiny raw deviation, enormous in train-std units
    transformed = Standardizer.fit(train).transform(held_out)
    assert np.all(np.abs(transformed) < 15.0)


def test_svd_fallback_matches_numpy() -> None:
    from evoforest_arch.readout import _svd_via_gram, robust_singular_values

    x = np.random.default_rng(0).normal(size=(30, 8))
    x[:, 7] = x[:, 6]  # rank deficient, like duplicated output blocks
    for matrix in (x, x.T):
        u, s, vt = _svd_via_gram(matrix)
        keep = s > 1e-8
        np.testing.assert_allclose((u[:, keep] * s[keep]) @ vt[keep], matrix, atol=1e-8)
        np.testing.assert_allclose(robust_singular_values(matrix), np.linalg.svd(matrix, compute_uv=False), atol=1e-8)
