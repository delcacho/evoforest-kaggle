# Kaggle House Prices with EvoForest

Applies this repo's graph-evolution substrate to
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
The leaderboard metric is RMSE on log(SalePrice), so the target is `log1p(SalePrice)`
and every RMSE printed here is on the leaderboard's scale.

For the experiment report (code changes, the overnight local-Qwen run and the Kaggle
result), see [docs/house_prices_experiment.md](../../docs/house_prices_experiment.md).

The whole pipeline also runs as one script: `STEPS=200 examples/kaggle_house_prices/run_all.sh`
(see the tunables at the top of that file).

## 1. Get the data

```bash
pip install kaggle   # needs ~/.kaggle/kaggle.json and the competition rules accepted
kaggle competitions download -c house-prices-advanced-regression-techniques -p examples/kaggle_house_prices/data/raw
unzip examples/kaggle_house_prices/data/raw/*.zip -d examples/kaggle_house_prices/data/raw
```

## 2. Prepare features

```bash
python examples/kaggle_house_prices/house_prices.py prepare
```

This writes the following to `data/prepared/`:

- `house_prices_train.npz` / `house_prices_test.npz`: one strictly positive float matrix
  `x`, scaled as `1 + (v - train_min) / train_std` (unseen test rows are clipped at 0.1). Missing values are filled using their meaning in the data description,
  quality grades are ordinal-encoded, skewed areas get `log1p`, and categoricals
  are one-hot encoded.
- `dataset_manifest.json`: an `external-manifest` that `evoforest-arch evolve` can load.
- `task_context.md`: a domain brief mapping each column index to its feature name,
  passed to the LLM agents via `--task-context-file`. It also holds a column profile
  computed on training rows: type, skew, correlation, a text histogram and modality.

All imputation and scaling statistics come from training rows only. The two
partial-sale outliers (GrLivArea > 4000 and price < $300k) are dropped by default;
pass `--keep-outliers` to keep them.

Columns are sorted by |corr(feature, y)|. The seed graph's interaction primitives
only look at the first few columns, so this sorting makes them pair OverallQual,
TotalSF and GrLivArea.

## 3. Baseline (no LLM)

```bash
python examples/kaggle_house_prices/house_prices.py baseline
```

This takes the tabular seed graph and picks the best configuration by 10-fold CV RMSE.
It then refits on all rows and writes `submissions/baseline.csv`. On the Kaggle
train data it reaches a CV RMSE of about 0.110.

## 4. Evolve the graph (needs an LLM key in `.env`)

```bash
evoforest-arch evolve \
  --dataset external-manifest \
  --dataset-manifest examples/kaggle_house_prices/data/prepared/dataset_manifest.json \
  --task-context-file examples/kaggle_house_prices/data/prepared/task_context.md \
  --scorer rmse --folds 10 --steps 40 \
  --island-devices cpu:0,cpu:1,cpu:2,cpu:3 \
  --promotion-policy cv_sign_test --sign-test-alpha 0.05 --max-validation-regression 0.10 \
  --allow-source-mutations \
  --output runs/house-prices
```

On a CUDA box, drop `--island-devices` to use `cuda:0..3` with L-BFGS refinement.
Resume with `--resume --steps N`.

`cv_sign_test` accepts a mutation only if it beats the incumbent on a significant
number of CV folds: 9 of 10 at α = 0.05. It also vetoes candidates whose validation
score is more than 10% worse. Each decision is logged to
`runs/house-prices/promotion_decisions.jsonl`.

`evolve` splits Kaggle's train.csv into its own train, validation and test splits.
`evoforest-arch recheck runs/house-prices --include-test` reports the internal
holdout score.

## 5. Submission from the evolved graph

```bash
python examples/kaggle_house_prices/house_prices.py predict runs/house-prices --cv --allow-source
```

This loads `best_graph.json` and `best_config` from the run, refits the readout on
*all* Kaggle training rows, and writes `submissions/evolved.csv`.

- `--cv` re-scores the graph on the full train set.
- `--allow-source` is required when the graph contains LLM-written lambdas.

Some primitives use batch statistics such as column means. To keep the feature
transform identical for train and test rows, the graph is evaluated once on the
stacked train+test matrix. The target is never used.

```bash
kaggle competitions submit -c house-prices-advanced-regression-techniques \
  -f examples/kaggle_house_prices/submissions/evolved.csv -m "EvoForest evolved graph"
```
