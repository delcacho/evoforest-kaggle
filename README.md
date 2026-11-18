# EvoForest Reimplementation

> Based on [Gabriel Kahen's evoforest-reimplementation](https://github.com/Gabriel-Kahen/evoforest-reimplementation) (MIT).
> This fork adds a fold sign-test acceptance policy, support for local OpenAI-compatible LLMs
> (e.g. Qwen on vLLM or llama.cpp), readout robustness fixes and a Kaggle House Prices experiment.
> See [docs/house_prices_experiment.md](docs/house_prices_experiment.md).

This repo recreates the software architecture described in
the EvoForest paper. It is not a reproduction of the authors' private evolved graph,
their 600-step run, or their reported score.

See the paper here: https://arxiv.org/abs/2604.19761

## What this fork adds

The rest of this README documents the original architecture. This section lists what
was added on top of it. The full experiment report, with results and lessons, is in
[docs/house_prices_experiment.md](docs/house_prices_experiment.md).

### 1. Sign-test promotion policy (`--promotion-policy cv_sign_test`)

The paper-style `cv_score` policy promotes a candidate whenever its mean CV score
improves, which over hundreds of steps accumulates noise. `cv_sign_test` instead:

- compares candidate and incumbent **fold by fold**, on the same fold assignment;
- drops ties and runs a **one-sided binomial sign test**. With 10 folds and
  `--sign-test-alpha 0.05` that means winning at least 9 of 10 folds; 0.10 means 8/10;
- **vetoes** candidates whose validation score is more than `--max-validation-regression`
  worse than the incumbent's (default 0.10; `<= 0` disables). The veto also applies to
  `cv_score`, except in `--profile paper`, which keeps the paper's no-veto behaviour;
- appends every decision (wins, losses, p-value, train/validation deltas, veto) to
  `<run>/promotion_decisions.jsonl`.

```bash
evoforest-arch evolve ... --folds 10 --promotion-policy cv_sign_test \
  --sign-test-alpha 0.05 --max-validation-regression 0.10
```

The fold p-value is optimistic because folds share training rows. Use at least about
9 folds, or significance is unreachable.

### 2. Local OpenAI-compatible LLMs

Any OpenAI-compatible server works: llama.cpp `llama-server`, vLLM, Ollama and similar.
No API key is needed for a non-OpenAI base URL. Example `.env`:

```bash
EVOFOREST_LLM_PROVIDER=openai
EVOFOREST_OPENAI_BASE_URL=http://127.0.0.1:8080/v1
EVOFOREST_LLM_MODEL=qwen
EVOFOREST_LLM_TIMEOUT_SECONDS=600
# Extra JSON merged into every request, e.g. disable Qwen3 thinking:
EVOFOREST_OPENAI_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}
# Optional engineer-only overrides (e.g. let only the code-writing agent think):
# EVOFOREST_ENGINEER_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": true}}
# EVOFOREST_ENGINEER_MAX_TOKENS=16384
EVOFOREST_LLM_INPUT_PRICE_USD_PER_MILLION_TOKENS=0
EVOFOREST_LLM_OUTPUT_PRICE_USD_PER_MILLION_TOKENS=0
```

Inline `<think>...</think>` blocks are stripped from responses.

### 3. More forgiving handling of LLM-written mutations

| Change | Why |
|---|---|
| Standard block YAML is parsed with PyYAML first (new dependency `pyyaml`), then the original line parser | Models emit valid multi-line YAML that the one-level parser rejected |
| Declared `n_columns` is advisory in production runs (width is taken from the actual output) | Models copied `n_columns: 1` from the prompt example while returning whole matrices |
| `FeatureBlock(...)` inside lambdas tolerates missing or miscounted names | Names are only diagnostic labels |
| An invalid optional `torch_source` falls back to auto-derivation instead of rejecting the feature | The torch path is only used by global refinement |
| New intermediate nodes that nothing consumes are auto-wired into the output as pass-through features | Otherwise the idea is pruned as unreachable and scored as "no change" |
| `combine_blocks` names multi-column arrays per column | Fixes width mismatches for multi-column lambdas |
| Sandbox allow-list gained pure elementwise calls (`log1p`, `sqrt`, `tanh`, `where`, `power`, `exp`, ...), and the allow-list is printed in the engineer prompt | Fewer avoidable sandbox rejections |
| Sandbox memory limit raised from 512 MB to 8 GB | `RLIMIT_AS` counts virtual address space; NumPy/BLAS reserve a lot of it |
| Engineer prompt includes the task context, graph semantics (node alternatives compete; output alternatives stack) and "use the column map, never guess indices" | Cuts structurally wrong mutations |

### 4. Readout robustness

- **Extrapolation clipping.** Held-out standardized features are clipped to the training
  range ±3 (`readout.EXTRAPOLATION_MARGIN`). Columns that are constant relative to their
  mean are treated as constant. Saturated gates had produced held-out values in the
  thousands of standard deviations.
- **SVD fallback.** `_robust_svd` tries NumPy, then SciPy `gesvd`, then a Gram-matrix
  eigendecomposition, so rank-deficient matrices no longer crash with
  `SVD did not converge`.

### 5. Kaggle House Prices example

[examples/kaggle_house_prices/](examples/kaggle_house_prices/README.md) contains:

- **`house_prices.py prepare`**: cleaning, ordinal maps, engineered features (TotalSF,
  ages, baths, porch area, `Has*` flags), one-hot encoding, and strictly positive
  scaling. It also writes a task context with a per-column profile (type, skew,
  correlation, text histogram, modality) so the LLM can reason about distributions.
- **`baseline`**: the seed graph only, no LLM.
- **`predict`**: refit an evolved graph on all training rows and write a submission CSV.
- **`run_all.sh`**: the end-to-end pipeline, configurable through environment variables
  (`STEPS`, `FOLDS`, `PROMOTION`, `SIGN_TEST_ALPHA`, `LLM_URL`, `START_LLM`,
  `SUBMIT`, ...).

The Kaggle data is not included; the competition rules forbid redistributing it.

### 6. Tests

`tests/test_promotion_sign_test.py` covers the sign test, the veto, advisory widths,
extrapolation clipping and the SVD fallback. `tests/test_kaggle_house_prices.py`
covers the example's preparation and prediction path on synthetic frames, so no Kaggle
data is needed. The example needs pandas: `pip install -e '.[dev,kaggle]'`. Without it,
those tests are skipped.

Run everything with `pytest -q`.

The implementation focuses on the reusable system design:

- A shared directed acyclic graph of computational nodes.
- A task schema that declares row-aligned inputs and selects an appropriate seed
  graph/primitive registry for the current task.
- A dataset loader registry for built-in smoke datasets, `.npz` files, JSON
  manifests, and Python module hooks.
- Intermediate nodes with multiple alternative implementations.
- Callable-family nodes for reusable projections, gates, and activations.
- Persistent low-dimensional global parameters.
- A single output node whose alternatives are all evaluated as ensemble features.
- Fitting nodes (`ridge_w`, `ridge_g`) that alter sample weighting and iterative
  residual reweighting.
- A Ridge-based cross-validation evaluator with deterministic coverage-aware capped
  configuration planning. Large plans use a clearly labeled, single-fold fixed-alpha
  approximate screen; only finalists enter the unchanged exact CV scorer, and the
  selected winner always passes that exact gate. Screening can exclude the true
  full-plan exact-CV optimum, so selection is exact among finalists rather than an
  exhaustive optimality guarantee.
- Task-aware fold strategies: random, grouped random, leave-group-out style
  grouped folds, stratified target folds, and time-blocked folds.
- Objective-aware scoring with raw metric direction preserved for lower-is-better
  losses such as RMSE and MAE while the optimizer still maximizes a score.
- Ancestor-conditioned subpath caching shared across evaluated configurations.
- A paper-style two-phase evaluation path: PyTorch L-BFGS global refinement is
  attempted by default when trainable globals are active, then globals are frozen
  for Ridge scoring. A deterministic NumPy coordinate backend remains available
  only when explicitly requested for compatibility experiments.
- Structured diagnostics plus TOON-like reports with scoring context, feature
  correlations, residual signals, exact additive linear contribution summaries,
  a diagnostic global Ridge fit, effective rank, and fold stability.
- Persistent alternative-level age and quality summaries accumulated from the
  best evaluated configuration's feature/dependency diagnostics.
- Required OpenAI, Claude, or Gemini LLM scientist, engineer, and memorandum
  agents that use paper-style prompt artifacts. LLM mutation synthesis is
  lambda-first and accepts paper-style node-keyed YAML such as `add: output:
  - "lambda ctx, values: ..."`. Shorthand lambdas infer parents from
  `values["parent"]`; the extended schema can also declare `parents`,
  `global_refs`, `node_kind`, `output_contract`, and `torch_source`. In island
  mode the scientist agent defaults to
  the paper's fixed temperature schedule `(0.35, 0.5, 0.6, 0.75)`, while
  engineer synthesis and memorandum updates default to temperature `0`.
- Cached task-context summaries with tensor inventory, target summary, scorer
  mechanics, and implementation constraints injected into LLM prompts.
- Node-level mutation support so a document can introduce a new intermediate,
  callable, output, or fitting node before adding alternatives to it.
- Optional sandboxed source-backed mutation alternatives that store and execute
  paper-style `lambda ctx, values: ...` implementations from mutation YAML with
  timeout, resource-limit, deterministic rerun, and output-contract checks.
  Feature-producing NumPy lambdas get a simple automatic torch expression when
  one can be safely derived; callable and fitting nodes use sandbox-safe
  declarative source specs.
- Graph maintenance for duplicate collapse, unreachable pruning, and unused globals.
- Failed generated candidates are rejected, logged into events/memoranda, and fed
  back to the next engineer prompt.
- Rejected but executable candidates still go through salvage for locally useful
  alternatives.
- A mutation/evolution loop with persistent JSON artifacts, sectioned
  hypothesis-free memoranda, a versioned global-best archive, sequential island
  mode, demo thread-backed islands, and production process-backed island actors.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

For the optional PyTorch L-BFGS refinement backend:

```bash
python -m pip install -e '.[dev,torch]'
```

## Run Tests

```bash
pytest -q
```

## Run Demo

```bash
evoforest-arch demo --steps 12 --islands 2 --async-islands --refine-backend auto --output runs/demo
```

## Run Production Evolution Smoke

The production workflow is the safer path for serious graph search. It writes a
run manifest, dataset fingerprint, fixed train/validation/test split manifest,
resume state, reloadable graph artifacts, and a stricter archive that only promotes
graphs after both train improvement and validation recheck improvement. By
default, production `evolve` uses the paper-native four-island asynchronous
topology with dedicated devices `cuda:0,cuda:1,cuda:2,cuda:3`.

```bash
evoforest-arch evolve --steps 4 --seed 17 --n-series 240 --length 160 --output runs/production-smoke
evoforest-arch evolve --dataset synthetic-tabular --steps 4 --seed 17 --n-samples 240 --n-features 12 --output runs/production-tabular-smoke
evoforest-arch evolve --dataset synthetic-tabular --scorer rmse --fold-strategy stratified --steps 4 --output runs/rmse-stratified-smoke
evoforest-arch inspect runs/production-smoke
evoforest-arch export-best runs/production-smoke --output runs/production-smoke-best.json
evoforest-arch recheck runs/production-smoke
```

Each production island runs as its own OS process actor and owns proposal,
repair, evaluation, prompt records, state, and memorandum updates. Candidate
commits and global-best migrations are sent to the target island as actor
messages, and the coordinator replaces its scheduling view from the actor's
returned snapshot. The root run directory is the global-best ledger. To run a
local smoke test without CUDA, pass explicit CPU-like device slots:

```bash
evoforest-arch evolve --steps 4 --island-devices cpu:0,cpu:1,cpu:2,cpu:3 --no-refine-globals --output runs/production-cpu-smoke
```

For the closest clean-room match to the paper's reported long run, use the paper
profile. It sets 600 target steps, four async islands, dedicated `cuda:0..3`
devices, 64 configurations, PyTorch L-BFGS refinement, scientist temperatures
`0.35,0.5,0.6,0.75`, engineer temperature `0`, and CV task score promotion without
the production validation gate:

```bash
evoforest-arch evolve --profile paper --llm-provider env --env-file .env --output runs/paper-profile
```

To continue an interrupted run, pass `--resume`; `--steps` then means additional
steps:

```bash
evoforest-arch evolve --resume --steps 4 --output runs/production-smoke
```

The test split is not evaluated during `evolve` or default `recheck`. Consume it
only when you are ready to burn the holdout:

```bash
evoforest-arch recheck runs/production-smoke --include-test
```

The built-in `synthetic-structural-break` and `synthetic-tabular` datasets are
workflow smoke tests. Real tasks can be loaded through the external dataset
interfaces. For a row-aligned `.npz` file:

```bash
evoforest-arch evolve \
  --dataset external-npz \
  --dataset-path data/task.npz \
  --target-key y \
  --input-key features \
  --input-key engine_id \
  --task-schema-file data/task_schema.json \
  --output runs/external-task
```

Manifest files can keep paths relative to the manifest:

```json
{
  "adapter": "external-npz",
  "path": "task.npz",
  "target_key": "y",
  "input_keys": ["features", "engine_id"],
  "task_schema": {
    "name": "external-tabular",
    "kind": "tabular",
    "inputs": [
      {"name": "features", "kind": "numeric_matrix", "roles": ["feature"]},
      {"name": "engine_id", "kind": "group_id", "roles": ["group", "unit"]}
    ],
    "default_input": "features"
  }
}
```

Run it with:

```bash
evoforest-arch evolve --dataset external-manifest --dataset-manifest data/dataset_manifest.json --output runs/external-manifest
```

Python hooks are also supported with `--dataset python-module --dataset-module
package.module --dataset-function load_dataset`. When a manifest uses
`"adapter": "python-module"`, the manifest directory is temporarily added to
`sys.path` while importing and calling the loader, so local loader modules can
live next to the manifest. A real strength claim still needs a fixed dataset
version, fixed split manifest, and no tuning on the held-out test split; see
[docs/evolution_workflow.md](docs/evolution_workflow.md).

## Run Benchmarks

The benchmark suite generates reproducible JSON and Markdown reports that show
architecture-level faithfulness to the paper without claiming to reproduce the
authors' private evolved graph or reported score:

```bash
python -m benchmarks.run_all --quick --output benchmark_reports/quick
```

See [docs/benchmarks.md](docs/benchmarks.md) for the full benchmark methodology.

For the leakage-aware generalization study with controlled DAG discovery and
optional frozen external regression
manifests, see [docs/research_study.md](docs/research_study.md).

Evolution always uses the paper-style LLM scientist, engineer, and memorandum
pipeline. The repository is prepared for Gemini; copy the checked-in example:

```bash
cp .env.example .env
```

```dotenv
EVOFOREST_LLM_PROVIDER=gemini
GEMINI_API_KEY=...
EVOFOREST_LLM_MODEL=gemini-3.1-flash-lite
```

Optional shared settings:

```dotenv
EVOFOREST_LLM_TIMEOUT_SECONDS=120
EVOFOREST_LLM_MAX_TOKENS=8192
EVOFOREST_LLM_MAX_RETRIES=0
EVOFOREST_LLM_TARGET_BUDGET_USD=5
EVOFOREST_LLM_HARD_BUDGET_USD=20
EVOFOREST_LLM_INPUT_PRICE_USD_PER_MILLION_TOKENS=0.25
EVOFOREST_LLM_OUTPUT_PRICE_USD_PER_MILLION_TOKENS=1.50
```

Validate the configuration without making an API call, then run:

```bash
evoforest-arch llm-check --env-file .env
evoforest-arch demo --steps 4 --llm-provider env --env-file .env --output runs/llm-demo
```

The paper-style island scientist schedule can be overridden with
`--llm-island-temperatures 0.35,0.5,0.6,0.75`. Use
`--llm-island-temperatures none` to reuse `--llm-scientist-temperature` for every
island.

By default, runs write a deterministic `task_context.md` from the runtime inputs
and evaluator settings. To inject a hand-authored or externally generated domain
brief into LLM prompts, pass `--task-context-file path/to/context.md`.

`ridge_g` residual rules run as iterative reweighted least squares. Use
`--irls-steps` to control the number of residual-weighted refits after the initial
Ridge fit.

LLM-generated mutation documents may include source-backed lambda alternatives by
default, matching the paper's mutation representation. Source lambdas are
AST-validated and evaluated in a subprocess
sandbox with timeout, memory limit where supported, deterministic repeated
evaluation, and output-contract checks. Source alternatives can target
feature-producing, callable, and fitting nodes; callable/fitting source returns a
declarative spec that is materialized by the parent process. This is a practical
local execution boundary, not a replacement for an OS/container sandbox for
hostile code.

LLM mode is fail-fast. If the configured provider is missing, the HTTP request
fails, the scientist response cannot be parsed into hypotheses, or the engineer
response cannot be parsed and validated as a mutation document, the run raises an
error without substituting another proposal mechanism. Prompt artifacts written
before the error are kept under the run's `prompts/` directory.

The production workflow can use the same `.env` wiring:

```bash
evoforest-arch evolve --steps 4 --llm-provider env --env-file .env --output runs/production-llm
```

The demo generates synthetic structural-break data, builds a seed graph, evaluates
graph configurations with a Ridge readout, derives scientist/engineer mutation
documents from diagnostics, and writes mutation documents, events, checkpoints, TOON
diagnostics, versioned global-best archive entries, prompt records when LLM agents
are enabled, execution-error records, task-context summaries, and memoranda under
the chosen output directory. Memoranda use the paper-style sections
`[OUTCOME HISTORY]`, `[STATE]`, `[WHAT WORKS]`, `[WHAT FAILED]`, and
`[ERROR LOG]`.

## Scope

The core library intentionally excludes competition-specific code, hidden benchmark
logic, and private target-specific feature selection. The one exception is the
self-contained Kaggle example under `examples/`, which this fork adds. It is meant as a clean architecture substrate
for experiments with open-ended computational graph evolution.

It also does not claim to recover the private 600-step evolved graph. The global
refinement phase attempts PyTorch L-BFGS for differentiable primitives and
source alternatives with provided or safely auto-derived torch expressions. It
records an explicit skipped-refinement reason when the torch path is unavailable
or a gradient probe finds no active trainable global influence. The NumPy
coordinate refiner is retained as an explicit compatibility backend, not as the
paper-mode fallback.
Production `evolve` is island-native by default: four persistent island process
actors own their graph state and are assigned one dedicated device each
(`cuda:0,cuda:1,cuda:2,cuda:3` unless `--island-devices` overrides them). It
persists per-island state, graph artifacts, checkpoints, memoranda, job logs, and
migration records from inside the island actor, so resume restores the island
frontier instead of restarting from a seed graph. It preserves the paper's fixed
four-temperature scientist schedule by default. The `--profile paper` preset switches promotion from the
default held-out validation gate to the paper's CV task score frontier and records
that contract in the manifest. The scientist/engineer/memorandum loop always uses
the configured LLM provider.
It does not include the authors' private model, exact prompts, full private
code-generation backend, or cluster scheduler.
The source-backed mutation path recreates the paper's
lambda-alternative representation for trusted local experiments, but it is not a
security sandbox. Alternative-level age and
quality history is implemented as rolling clean-room summaries over participating
alternatives in accepted stateful evaluations, not as the authors' private statistic
schema. Linear SHAP-style diagnostics are exact additive decompositions of this
repo's standardized Ridge readout (`z_j * coefficient_j`); they are not a claim to
match a private SHAP implementation byte-for-byte. Cross-configuration caching
assumes alternatives are deterministic functions of their parents, inputs, and
globals during one evaluator pass. The graph semantics, configuration scoring,
fitting-node hooks, diagnostics, mutation artifacts, graph maintenance,
failed-mutation feedback, salvage behavior, task-context summaries, prompt records,
global-best archive, and memoranda mirror the paper's software architecture.
Task-context summaries are
deterministic summaries of runtime inputs and scorer mechanics rather than private
LLM-authored domain briefs.
