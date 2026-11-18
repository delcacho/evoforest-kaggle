# EvoForest + a local LLM on Kaggle House Prices

This is an experiment report. It covers what was changed in this reimplementation to run
the EvoForest-style search (an LLM proposes feature-graph mutations, which are scored
through a Ridge readout) against a local Qwen model on Kaggle's
[House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
competition. It also covers what happened when we did.

**Short version:**

- Replacing "accept any CV improvement" with a fold-level sign test was the most useful
  change.
- The LLM found a few sensible, textbook features.
- On the leaderboard, the evolved graph and the hand-built seed graph are practically
  indistinguishable. On this dataset the score depends more on how a single extreme test
  house is handled than on feature discovery.

## 1. Setup

| Item | Value |
|---|---|
| Data | Kaggle `train.csv`, 1,460 houses; 1,458 after dropping the two well-known partial-sale outliers (GrLivArea > 4000, price < $300k) |
| Target | `log1p(SalePrice)`; all RMSEs below are on the leaderboard's log scale |
| Features | 79 raw columns turned into 206: 143 one-hot, 53 numeric/ordinal, 10 engineered (TotalSF, TotalBath, TotalPorchSF, HouseAge, RemodAge, Has* flags) |
| Search splits | 874 train / 292 validation / 292 test (internal, from Kaggle's train.csv) |
| Fitness | 10-fold CV RMSE on the 874 train rows |
| LLM | Local Qwen behind an OpenAI-compatible server (vLLM / llama.cpp), no cloud calls |
| Readout | Ridge, with the seed graph's alternatives (raw / abs / square / interactions, activations, summary stats) chosen by configuration search |

Everything is in `examples/kaggle_house_prices/`. `run_all.sh` runs the whole pipeline:
data, preparation, LLM check, baseline, evolution and submission.

## 2. What was changed

### 2.1 Acceptance: a fold sign test instead of "mean CV went up"

The paper-style policy (`cv_score`) accepts a mutation whenever the mean CV score
improves. Over hundreds of steps this turns into hill-climbing on noise. In our logs,
roughly a fifth of all proposals improved both train CV and validation by tiny amounts,
and a "both improved" rule would have accepted every one of them.

The new policy `cv_sign_test` (`evoforest_arch/production.py`, `fold_sign_test`) works
like this:

- It compares the candidate and the incumbent fold by fold, on the same fold assignment.
- Ties are dropped, and it runs a one-sided binomial sign test (`--sign-test-alpha`,
  default 0.05).
- With 10 folds that effectively means **winning at least 9 of 10 folds** (p = 0.011).
- A **validation regression veto** (`--max-validation-regression`, default 10%) also
  rejects candidates that fall apart on unseen rows.
- Every decision is appended to `promotion_decisions.jsonl` in the run directory: wins,
  losses, p-value, train and validation deltas, and veto status.

**Caveat:** folds share training rows, so the p-values are optimistic. It is still far
stricter about lucky averages than a mean comparison.

### 2.2 Numerical robustness of the readout

| Problem | Fix |
|---|---|
| On positive inputs, `sigmoid_gate` saturates into near-constant columns. Their tiny train std then blew up standardized values on held-out rows (one fold scored RMSE ≈ 29). | Columns that are constant relative to their mean are treated as constant, and held-out standardized values are clipped to the train range ± 3 (`readout.py`, `EXTRAPOLATION_MARGIN`). |
| `SVD did not converge` on rank-deficient feature matrices | `_robust_svd` tries NumPy first, then SciPy `gesvd`, then an SVD computed from the Gram matrix. |

### 2.3 Making a local LLM workable

- **OpenAI-compatible local endpoints.** The client accepts:
  - `EVOFOREST_OPENAI_BASE_URL`, with no API key needed for local servers;
  - `EVOFOREST_OPENAI_EXTRA_BODY`, e.g. to turn Qwen3 thinking off;
  - separate engineer settings (`EVOFOREST_ENGINEER_EXTRA_BODY`,
    `EVOFOREST_ENGINEER_MAX_TOKENS`).

  Inline `<think>` blocks are stripped from responses. See `.env.example`.
- **More forgiving parsing of what the LLM writes:**
  - Ordinary block-style YAML is parsed with PyYAML before falling back to the original
    line parser.
  - Declared column counts are advisory, because the model copied `n_columns: 1` from the
    prompt example while returning whole matrices.
  - Miscounted `FeatureBlock` names become generic labels.
  - An invalid optional `torch_source` no longer sinks a valid NumPy feature.
- **Auto-wiring.** New intermediate nodes that nothing consumes are wired into the output
  as pass-through features. Without this, the LLM's most common mistake (declaring a
  feature but not routing it) made ideas silently disappear and score as "no change".
- **Sandbox.**
  - A handful of pure elementwise NumPy/torch calls were added to the allow-list (`log1p`,
    `sqrt`, `tanh`, `where`, ...), and the allow-list is shown in the engineer prompt.
  - The memory limit was raised to 8 GB, because `RLIMIT_AS` counts virtual address space
    and NumPy/BLAS reserve a lot of it.
- **Engineer prompt.** It now includes the task context and explains the graph semantics:
  alternatives inside a node compete, while output alternatives are stacked.

### 2.4 Giving the LLM distribution awareness

The task context includes a **column profile** computed on training rows only, with one
row per column:

- type (binary / discrete / continuous), number of distinct values and share at the
  minimum;
- median, p99, max, skew and correlation with the target;
- a 12-bin text histogram plus a uni/bi/multi-modal label.

Without this the model was proposing transforms blind. With it, it started reasoning
about zero-inflation and saturation.

### 2.5 Tests

`tests/test_promotion_sign_test.py` and `tests/test_kaggle_house_prices.py` cover:

- the sign-test thresholds, tie handling and veto;
- advisory column counts;
- extrapolation clipping and the SVD fallback;
- the House Prices preparation and prediction path.

## 3. The overnight run

The run used 10 folds, `cv_sign_test` at α = 0.05, the 10% validation veto and source
mutations allowed. It made 200 steps planned on one island, with no refinement. Step times
grew from about 4.5 to 7.5 minutes as accepted features enlarged the configuration search.

### Accepted mutations (first ~140 steps)

| Step | Feature proposed by the LLM | Folds won | Train CV | Validation |
|---|---|---|---|---|
| seed | – | – | 0.11171 | 0.12444 |
| 5 | `OverallQual × TotalSF` | 10/10 | 0.11112 | 0.12401 |
| 17 | `log(TotalSF) × OverallQual` (diminishing returns to area) | 10/10 | 0.11086 | 0.12384 |
| 65 | `HouseAge / (HouseAge + RemodAge)` (remodelled old stock) | 9/1 | 0.11073 | 0.12389 |

All three are sensible, and all three are things an experienced Kaggler would add in the
first hour.

### What the rejections looked like

- **Noise:** about 10–15% of proposals won 7 or 8 of 10 folds with train gains of
  0.00001–0.0005. That's exactly what the sign test is meant to stop.
- **Configuration-switch artefacts:** a recurring pattern of +0.001 on train CV with
  −0.004 on validation. The configuration search re-selects a slightly different seed
  setup, so the mutation itself isn't what is being measured.
- **Blow-ups:** a "centered square" and a tanh transform scored validation RMSE of
  0.4–0.6. These were rejected by the sign test and/or the validation veto.
- **Repetition:** the LLM kept cycling through five families:
  - quality × area;
  - basement quality × size (7/3 at least five times, never confirmed);
  - remodel age;
  - garage ratios (GarageArea / GarageCars, proposed about five times);
  - lot/area ratios.

  "Replace square with log" was proposed at least eight times. The memorandum records
  failures by the node name the LLM invented, not by formula, so near-duplicates aren't
  recognised.

## 4. Kaggle result

The evolved graph (snapshot at step ~131, three acceptances) was refit on all 1,458
training rows.

| Submission | 10-fold CV (all train rows) | Public LB |
|---|---|---|
| Seed graph, no LLM (`baseline.csv`) | 0.10977 | not submitted |
| Evolved graph, step ~131 (`evolved-v5-step125.csv`) | 0.10969 | **0.12998** |

Three observations:

1. **On the full data, the LLM's gain almost vanishes** (0.10977 → 0.10969). Refit on
   all rows, the seed's configuration search picks its built-in `interactions` option.
   Because columns are sorted by correlation, that option already multiplies
   OverallQual × TotalSF, the LLM's main discovery. The 0.9% improvement during the
   search was measured against a seed configuration that happened not to use
   interactions. The seed and evolved predictions differ by 0.016 in log-price SD.
2. **The gap from CV to the leaderboard (0.110 → 0.130) is mostly one house.** Test row
   Id 2550 is a 5,095 sq ft, quality-10 house in Edwards, and both models predict about
   $880–890k for it. It closely resembles the two partial-sale outliers dropped from
   training, which sold for under $200k. Having dropped them, a linear model with squares
   and quality × area products extrapolates straight up. If its true price is similarly
   low, that single row costs on the order of 0.01 RMSE. Whether to keep those training
   outliers, or to cap extrapolation, matters more here than any feature found.
3. **Rank means little on this leaderboard.** The House Prices public leaderboard is
   crowded with leaked near-zero scores. Good linear-model notebooks typically land at
   0.117–0.120.

## 5. Conclusions

- **The sign test is the important design choice.** A significance test on fold-level
  wins, plus a validation veto, turned a noise-accumulating hill-climber into a search
  that accepted 3 of ~140 proposals, each with a clear rationale. The same idea works as a
  filter for brute-force feature generators: it is related to Boruta's binomial test
  against shadow features.
- **The LLM behaved like a careful junior data scientist.** It produced plausible
  hypotheses grounded in the domain and the column profile, and the ones that passed
  were textbook. It did not find anything non-obvious on this dataset, and it wasted a lot
  of steps on repeats.
- **A strong seed leaves little headroom.** Most of the score comes from conventional
  preparation (TotalSF, ages, ordinal quality scales, log transforms) and a regularised
  Ridge with interactions.
- **Feature search can't see data problems.** The search only sees aggregate fold scores,
  so it could never notice the extreme test house or reason about the dropped outliers.

## 6. What I would change next

1. **Log attempts by formula, not by name,** with the sign-test result attached (e.g.
   `GarageArea/(GarageCars+1): 2/8 folds`). Also skip exact duplicates by comparing
   normalized source code before evaluating.
2. **Score candidates with the incumbent's configuration fixed,** so that a mutation is
   not confounded with the configuration search switching seed options.
3. **Use a more powerful paired test:** per-row out-of-fold errors (about 870 pairs
   instead of 10 folds) with a minimum effect size, or 5×10 repeated CV with a corrected
   t-test. Several "7/3" candidates could then be resolved instead of being rejected for
   lack of power.
4. **Give the LLM residual diagnostics** (worst-predicted training rows, computed on train
   folds only), so it can see outliers and extrapolation risk.
5. **Hide validation scores from the LLM** and use validation only for the veto, so it
   stays a cleaner holdout.
6. **Check transfer to trees:** feed the accepted features to LightGBM and blend it with
   the Ridge model.

## 7. Reproducing

```bash
# Local Qwen on an OpenAI-compatible endpoint, configured in .env (see .env.example)
STEPS=200 examples/kaggle_house_prices/run_all.sh

# Or step by step
python examples/kaggle_house_prices/house_prices.py prepare
python examples/kaggle_house_prices/house_prices.py baseline --folds 10
evoforest-arch evolve --dataset external-manifest \
  --dataset-manifest examples/kaggle_house_prices/data/prepared/dataset_manifest.json \
  --task-context-file examples/kaggle_house_prices/data/prepared/task_context.md \
  --scorer rmse --folds 10 --steps 200 --islands 1 --island-devices cpu:0 \
  --no-refine-globals --promotion-policy cv_sign_test --sign-test-alpha 0.05 \
  --max-validation-regression 0.10 --allow-source-mutations \
  --llm-provider env --env-file .env --output runs/house-prices
python examples/kaggle_house_prices/house_prices.py predict runs/house-prices --cv --allow-source
```

Results vary between runs: LLM sampling is stochastic, and the accepted features depend
on the order of proposals.
