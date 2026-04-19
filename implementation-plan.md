# Implementation Plan: ML-Orchestrator
## Version: 1.0
## Date: April 3, 2026

---

## Overview

This plan breaks the MVP into 10 sequential phases. Each phase is independently buildable and testable before moving to the next. Phases 1–3 establish the skeleton; Phases 4–9 are the ML workflow stages; Phase 10 wires everything together and polishes.

---

## Proposed Project Structure

```
ml_orchestrator/
├── main.py                  # entry point, main menu loop
├── config/
│   └── defaults.py          # sensible per-stage defaults (not state)
├── state/
│   └── manager.py           # project state: load, save, update (YAML)
├── ui/
│   ├── menus.py             # rich panels, questionary prompts, navigation
│   └── charts.py            # plotext inline chart helpers
├── stages/
│   ├── ingestion.py
│   ├── profiling.py
│   ├── cleaning.py
│   ├── feature_prep.py
│   ├── feature_selection.py
│   ├── modeling.py          # loads available models from registry
│   ├── tuning.py            # uses model's declared param grid
│   └── evaluation.py        # uses model's declared metrics
├── formats/
│   ├── base.py              # BaseFormat interface
│   ├── registry.py          # auto-discovers all formats by extension
│   ├── csv_format.py
│   └── parquet_format.py    # add new formats here — registry picks them up automatically
├── models/
│   ├── base.py              # BaseModel interface (see Model Registry section)
│   ├── registry.py          # discovers and registers all models in this dir
│   ├── classification/
│   │   ├── logistic_regression.py
│   │   └── random_forest.py
│   └── regression/
│       ├── linear_regression.py
│       └── random_forest.py
├── artifacts/
│   └── manager.py           # export: dataset, reports, config, script
├── deps/
│   └── checker.py           # dependency check + user-approved install
└── utils/
    └── logger.py            # append-only run log
```

Each stage module exposes a single `run(state) -> state` function. The main menu calls the appropriate stage and saves state after each call.

The `models/` directory follows a **registry pattern** — adding a new algorithm means adding one file. No changes to the modeling, tuning, or evaluation stages are required. See the Model Registry section below.

---

## Model Registry

Each algorithm lives in `models/classification/` or `models/regression/` and implements the `BaseModel` interface from `models/base.py`:

```python
class BaseModel:
    name: str                  # display name shown in menu
    task_type: str             # "classification" or "regression"
    dependencies: list[str]    # pip packages required beyond core stack

    def get_estimator(self, params: dict)  # returns a sklearn-compatible estimator
    def default_params(self) -> dict       # params used when user accepts defaults
    def param_grid(self) -> dict           # search space for tuning stage
    def metrics(self) -> list[str]         # metric names this model supports
    def artifacts(self) -> list[str]       # artifact types this model produces
                                           # e.g. ["confusion_matrix", "roc_curve", "feature_importance"]
```

`models/registry.py` auto-discovers all subclasses at startup. The modeling stage queries the registry for models matching the current `problem_type` and presents them as a menu.

**Adding a new algorithm** (e.g. XGBoost classifier):
1. Create `models/classification/xgboost.py` implementing `BaseModel`
2. Declare its `dependencies`, `param_grid`, `metrics`, and `artifacts`
3. The modeling, tuning, and evaluation stages pick it up automatically — no other files change

This keeps algorithm-specific knowledge (param grids, metrics, artifact types) co-located with the algorithm itself rather than scattered across stages.

---

## Phase 1: Project Scaffold + Entry Point

**Goal:** Runnable shell with a working main menu. No ML logic yet.

### Tasks
- [ ] Initialize repo structure with all directories and `__init__.py` files
- [ ] Set up `pyproject.toml` or `setup.py` with dependencies:
  - `rich`, `questionary`, `plotext`, `typer`, `pandas`, `scikit-learn`, `pyyaml`, `joblib`
- [ ] Implement `main.py` with `typer` entry point
- [ ] Implement main menu using `rich` + `questionary` (all 13 items from PRD Section 18)
- [ ] Implement basic navigation: select stage, show placeholder, return to menu
- [ ] Implement `utils/logger.py`: append-only log file per project run

### Acceptance
- `python -m ml_orchestrator` or `mlo` (if installed) shows the main menu
- Selecting any option prints a "not yet implemented" stub and returns to menu
- Exit option works cleanly

---

## Phase 2: Project State + Project Management

**Goal:** Create and load projects. State persists across sessions.

### Tasks
- [ ] Design project state schema (YAML). Fields per PRD 12.1:
  - `project_name`, `project_dir`, `created_at`, `dataset_path`
  - `schema`, `target_column`, `problem_type`
  - `cleaning_actions`, `selected_features`, `model_config`, `tuning_params`
  - `evaluation_summary`, `artifact_paths`, `dependency_log`
- [ ] Implement `state/manager.py`:
  - `create_project(name, directory) -> state`
  - `load_project(path) -> state`
  - `save_state(state)` — writes to `<project_dir>/state.yaml`
  - `update_state(state, **kwargs) -> state`
- [ ] Implement `config/defaults.py`: static defaults per stage (split ratio, random seed, etc.)
- [ ] Wire "Start New Project" and "Load Existing Project" menu items to state manager
- [ ] Handle keyboard interrupt during save gracefully (write to temp then rename)

### Acceptance
- Can create a project, exit, reload it, and state is preserved
- State file is human-readable YAML
- Missing or corrupt state file shows a clear error, does not crash

---

## Phase 3: Dependency Manager

**Goal:** Pre-stage dependency checking with user-gated installation.

### Tasks
- [ ] Implement `deps/checker.py`:
  - `check(packages: list[str]) -> list[str]` — returns missing packages
  - `prompt_and_install(missing: list[str], state) -> bool` — shows list, asks user approval, installs via `subprocess` + `pip`, logs to state
  - `install_in_venv(packages, venv_path)` — optional: install into project venv
- [ ] Each stage module declares its required packages at the top
- [ ] Main menu calls dependency check before entering any stage
- [ ] Log dependency actions (installed, skipped, failed) to state and run log

### Acceptance
- If a required package is missing, user is shown the list and asked before anything is installed
- If user declines, stage is skipped with a clear message
- If install fails, a clear error is shown and the stage is not entered

---

## Phase 4: Data Ingestion

**Goal:** Load CSV or Parquet, display schema summary, save to state.

### Tasks
- [ ] Implement `stages/ingestion.py`:
  - Prompt for file path using `questionary.path()` — tab-completion, path validation, re-prompt on invalid input
  - Load CSV (`pandas.read_csv`) or Parquet (`pandas.read_parquet`)
  - Display using `rich.Table`: row count, column count, inferred dtype per column
  - Show first 5 rows as a preview
  - Save `dataset_path` and inferred `schema` to state
- [ ] Handle common errors: file not found, unsupported format, parse errors
- [ ] Wire to "Read Dataset" menu item

### Acceptance
- Loads a CSV and Parquet file without errors
- Shows clean schema summary in the terminal
- Invalid path shows a useful error and re-prompts
- State contains `dataset_path` and `schema` after successful load

---

## Phase 5: Data Profiling

**Goal:** Surface data quality issues; show inline chart; optionally open HTML report.

### Tasks
- [ ] Implement `stages/profiling.py`:
  - Null count and null % per column
  - Unique count per column
  - Duplicate row count
  - Basic stats for numeric columns (min, max, mean, std)
  - Value frequency for low-cardinality categoricals
  - Warnings:
    - constant columns (unique == 1)
    - ID-like columns (unique == row count)
    - high-cardinality categoricals (unique > threshold)
    - severe class imbalance (if target known)
    - heuristic leakage risk (column name patterns, near-perfect correlation with target)
  - Display all via `rich` formatted output
- [ ] Implement `ui/charts.py` — `bar_chart(labels, values, title)` wrapper around `plotext`
- [ ] After profiling: display inline bar chart of top missing-value columns using `plotext`
- [ ] Prompt user to open HTML profiling report in browser:
  - If `ydata-profiling` available: generate report, open with `webbrowser.open(path)`
  - If not available: offer dependency install via Phase 3 checker
- [ ] Save profiling summary to state; export JSON summary to `<project_dir>/artifacts/`
- [ ] Wire to "Profile Dataset" menu item

### Acceptance
- Profiling output shows all required fields in readable format
- Warnings are visually distinct (e.g. yellow/red via `rich`)
- Inline bar chart renders in terminal after profiling
- User is prompted to open browser report; choosing yes opens it; choosing no continues
- Profiling can run without `ydata-profiling` (degrades gracefully)

---

## Phase 6: Data Cleaning

**Goal:** Let users handle missing values, duplicates, and column exclusions. Log decisions.

### Tasks
- [ ] Implement `stages/cleaning.py`:
  - Show missingness summary (from profiling state)
  - For each column with nulls: prompt strategy — drop column, drop rows, impute (mean/median/mode/constant), skip
  - Offer "remove duplicate rows" with confirmation
  - Offer "exclude columns from modeling" — multi-select via `questionary.checkbox`
  - **Outlier handling** — for each numeric column flagged as having outliers during profiling:
    - Detect using z-score (threshold: ±3), IQR (1.5×IQR), or MAD (robust, threshold configurable)
    - Show count and % of outlier rows per column
    - Prompt user for treatment: clip to boundary, remove rows, flag as new binary column, or skip
    - Log decisions to state (`cleaning_actions`)
  - **Class imbalance handling** — for classification tasks where target is known:
    - Show class distribution and imbalance ratio after target column is set
    - If ratio exceeds threshold (default: 5:1), warn user and offer treatment options:
      - Oversample minority class: `RandomOverSampler` (imbalanced-learn)
      - Undersample majority class: `RandomUnderSampler` (imbalanced-learn)
      - Synthetic oversampling: `SMOTE` (imbalanced-learn, requires optional install)
      - Use class weights: record `class_weight='balanced'` in state for modeling stage to apply
      - Skip (user handles imbalance at modeling time)
    - Sampling applied after train/test split to avoid leakage (only on train set)
    - Log decision to state (`imbalance_strategy`)
  - Show summary of all actions before applying
  - Apply actions to in-memory dataframe; save cleaned dataset to `<project_dir>/artifacts/cleaned_dataset.csv`
  - Log all decisions to state (`cleaning_actions`)
- [ ] Wire to "Clean Dataset" menu item

### Acceptance
- All cleaning choices are shown back to user before applying
- Cleaning actions are saved in state and reloadable
- Cleaned dataset is written to artifacts directory
- Outlier treatment options are shown only for columns flagged during profiling
- Class imbalance warning and treatment shown only for classification tasks
- SMOTE and sampler options degrade gracefully if `imbalanced-learn` is not installed (offer install via dependency manager)
- Skipping cleaning entirely is allowed (state records "no cleaning")

---

## Phase 7: Feature Preparation

**Goal:** Target selection, train/test split, encoding and scaling recommendations applied.

### Tasks
- [ ] Implement `stages/feature_prep.py`:
  - Prompt for target column selection from schema
  - Infer problem type (classification if target is categorical or low-cardinality int; regression otherwise)
  - Allow user to override inferred problem type
  - Prompt for test split ratio (default from `config/defaults.py`: 0.2)
  - Prompt for random seed (default: 42)
  - Prompt for stratification (classification only)
  - For each non-target column: recommend encoding (one-hot, ordinal, target encode) or scaling based on dtype
  - Supported scaling options:
    - None (no scaling)
    - Standard Scaling (Z-score): `StandardScaler()` — default recommendation for most algorithms
    - Zero-centering: `StandardScaler(with_mean=True, with_std=False)` — mean removal only
    - Range Scaling (Min-Max): `MinMaxScaler()` — for neural networks, KNN, ReLU activations
    - Robust Scaling: `RobustScaler()` — when data has significant outliers
    - Normalization (L2): `Normalizer(norm='l2')` — per-sample unit norm, useful for angle-based similarity
    - Whitening: `StandardScaler` + `PCA(whiten=True)` — decorrelates features, zero mean, unit variance
  - When user selects a scaler, show an algorithm-applicability hint (e.g. warn if Range Scaling is chosen for a tree-based model, as trees are scale-invariant)
  - Show recommendations as a table; allow user to override per column
  - Apply split and transformations; save transformer pipeline to state
  - Save `target_column`, `problem_type`, `split_config` to state
- [ ] Wire to "Prepare Features" menu item

### Acceptance
- Target column is selectable from a dropdown of column names
- Problem type inference is shown with reasoning; user can override
- Encoding/scaling recommendations are displayed and editable
- State contains all prep decisions after completion

---

## Phase 8: Feature Selection

**Goal:** Reduce features; show importance chart; save selected features to state.

### Tasks
- [ ] Implement `stages/feature_selection.py`:
  - Present method options via menu:
    1. Variance Threshold
    2. Univariate Selection (SelectKBest)
    3. Model-Based Importance (RandomForest feature importances)
    4. Recursive Feature Elimination (RFE with a simple estimator)
  - Prompt for method-specific parameters (threshold, k, etc.) with defaults
  - Run selected method; show:
    - Selected features (list)
    - Removed features (list)
    - Feature scores/rankings where available (table)
  - Display inline bar chart of feature scores using `plotext`
  - Save `selected_features` to state
- [ ] Wire to "Select Features" menu item

### Acceptance
- All four methods are available
- Feature scores chart renders in terminal
- Selected and removed features are clearly separated in output
- State contains `selected_features` after completion

---

## Phase 9: Baseline Modeling + Hyperparameter Tuning

**Goal:** Train a baseline model; optionally tune it; save model to artifacts.

### Tasks
- [ ] Implement `models/base.py` — `BaseModel` interface (see Model Registry section)
- [ ] Implement `models/registry.py` — auto-discovers all `BaseModel` subclasses in `models/classification/` and `models/regression/`
- [ ] Implement starter models (4 total):
  - `models/classification/logistic_regression.py`
  - `models/classification/random_forest.py`
  - `models/regression/linear_regression.py`
  - `models/regression/random_forest.py`
  - Each declares its `default_params`, `param_grid`, `metrics`, and `artifacts`
- [ ] Implement `stages/modeling.py`:
  - Confirm problem type from state (or re-prompt if missing)
  - Query registry for models matching `problem_type`; present as menu
  - Allow default params or prompt for basic manual overrides
  - Check model's `dependencies` via dependency manager before running
  - Train selected model; save to `<project_dir>/artifacts/model.joblib`
  - Save `model_config` (model name + params) to state
- [ ] Implement `stages/tuning.py`:
  - Load model's `param_grid()` from registry; display as YAML for user review/edit
  - Choose search type: GridSearchCV or RandomizedSearchCV
  - Run search with cross-validation (default cv=5)
  - Show best params and improvement over baseline
  - Save tuned model to `<project_dir>/artifacts/model_tuned.joblib`
  - Save `tuning_params` and results to state
- [ ] Wire to "Train Baseline Model" and "Tune Model" menu items

### Acceptance
- Classification and regression models both work
- Default param grids are provided so user can accept without typing
- Tuning shows best params and CV score vs baseline
- Both baseline and tuned models are saved as artifacts
- Adding a 5th model file to `models/classification/` makes it appear in the menu without any other changes

---

## Phase 10: Evaluation + Artifact Export

**Goal:** Display metrics; show inline chart; prompt for browser report; export all artifacts.

### Tasks
- [ ] Implement `stages/evaluation.py`:
  - Load model and test split from state
  - Look up the model in the registry; read its declared `metrics()` and `artifacts()`
  - Compute and display only the metrics the model declares (no hardcoded metric lists)
  - Generate artifacts the model declares (e.g. confusion matrix, ROC curve, residual plot)
  - Display all metrics via `rich` formatted panel
  - Display inline terminal bar chart of key metrics using `plotext`
  - Generate HTML evaluation report; prompt user to open in browser
  - Save evaluation summary dict to `<project_dir>/artifacts/evaluation_summary.json`
  - Save `evaluation_summary` to state
- [ ] Implement `artifacts/manager.py`:
  - `export_all(state)` — orchestrates export of:
    - cleaned dataset (already saved during cleaning)
    - profiling report (already saved during profiling)
    - workflow config (`workflow_config.yaml` — full state minus large objects)
    - selected feature list (`features.txt`)
    - evaluation summary (`evaluation_summary.json`)
    - run log
    - serialized sklearn Pipeline (`pipeline.joblib` — preprocessing + model in one object)
  - `generate_pipeline(state)` — build a sklearn `Pipeline` object from state decisions:
    - preprocessing steps (encoding, scaling) as named Pipeline steps
    - model as the final step
    - serialize with `joblib.dump(pipeline, "artifacts/pipeline.joblib")`
    - each step is commented in the export summary for readability
    - correctness guarantee: fit only on train split, transform applied consistently
- [ ] Wire to "Evaluate Model" and "Export Artifacts" menu items

### Acceptance
- Metrics display matches problem type
- Confusion matrix renders for classification tasks
- Inline metrics chart renders in terminal
- Browser report prompt works (opens or skips)
- `Export Artifacts` produces all expected files in `<project_dir>/artifacts/`
- Generated `pipeline_script.py` runs without modification on the same dataset

---

## Phase 11: Integration, Navigation, and Polish

**Goal:** Wire everything together; ensure stage-to-stage flow is coherent.

### Tasks
- [ ] Ensure main menu reflects stage completion status (e.g., checkmark next to completed stages)
- [ ] Enforce soft ordering: warn if user tries to run a stage before its prerequisite (e.g. modeling before feature prep), but do not block
- [ ] Ensure state is saved after every stage completes
- [ ] Ensure keyboard interrupt (`Ctrl+C`) at any stage exits cleanly without corrupting state
- [ ] Add "Settings" menu item: show/edit project defaults (split ratio, random seed, etc.)
- [ ] End-to-end test: run full workflow on a sample dataset (e.g. Iris, Boston housing equivalent)
- [ ] Review and clean up all terminal output for consistency (colors, spacing, section headers)

### Acceptance
- Full workflow runs from start to export on a real dataset without errors
- State file after full run contains all expected fields
- Reloading state and re-running evaluation produces the same result
- All exported artifacts are present and non-empty

---

## Build Order Summary

| Phase | Focus                          | Key Output                        |
|-------|--------------------------------|-----------------------------------|
| 1     | Scaffold + entry point         | Working menu shell                |
| 2     | State + project management     | Create/load projects              |
| 3     | Dependency manager             | Pre-stage dep checks + install    |
| 4     | Data ingestion                 | Load CSV/Parquet, schema display  |
| 5     | Profiling                      | Data quality warnings + charts    |
| 6     | Cleaning                       | Missing value + duplicate handling|
| 7     | Feature preparation            | Target, split, encoding, scaling  |
| 8     | Feature selection              | Importance chart, feature list    |
| 9     | Modeling + tuning              | Trained model, tuned model        |
| 10    | Evaluation + artifact export   | Metrics, charts, pipeline script  |
| 11    | Integration + polish           | Full end-to-end workflow          |

---

## Key Design Decisions

1. **Each stage is `run(state) -> state`** — uniform interface makes the main menu trivial and stages independently testable.

2. **State is YAML, written after every stage** — human-readable, easy to inspect, and safe to reload. Large objects (dataframes, models) are stored as file paths in state, not serialized into YAML.

3. **Dataframes are not stored in state** — state stores `dataset_path` and `cleaned_dataset_path`. Each stage that needs data loads it fresh. This keeps state files small and avoids stale in-memory data bugs.

4. **Plotext charts are non-blocking** — rendered inline and the workflow continues. The browser report prompt is the only optional async step.

5. **Dependency manager is called per-stage, not at startup** — users only need packages for the stages they actually use.

6. **Generated pipeline script mirrors state decisions** — it is built by reading state, not by recording code at runtime. This means it is always in sync with what was actually done.

---

## Dependencies (requirements.txt baseline)

```
rich>=13.0
questionary>=2.0
plotext>=5.0
typer>=0.9
pandas>=2.0
scikit-learn>=1.4
pyyaml>=6.0
joblib>=1.3
pyarrow>=14.0        # for Parquet support
ydata-profiling>=4.0 # optional, checked at runtime
```

---

## Open Items Before Coding Starts

- [x] Entry point name: `mlo`
- [x] Package management: `uv` — `pyproject.toml` + `uv.lock` committed
- [x] Venv: `uv`-managed `.venv` at repo root
- [x] Generated artifact format: sklearn `Pipeline` object, serialized with `joblib`, with clear step comments
- [x] GitHub: public repo, `main` / `dev` branches, merge to `main` at each phase completion
- [x] GitHub Actions: lint + tests on push to `dev` and `main`

---

## V2 Preview: Streamlit Companion UI (Post-V1)

Not implemented in V1. Listed here so V1 phase-level design choices (artifact formats, state schema, HTML report stubs) stay compatible with a future Streamlit layer and do not require rework.

### Scope
Streamlit + Plotly as on-demand companion surfaces for visual, exploratory stages. Launched from the terminal after a stage completes; replaces the static HTML report prompts in V1 Phases 5, 8, and 10.

### Stages with a Streamlit companion view
- **Profiling (Phase 5):** interactive column-level drill-down, filterable distributions, cross-column correlation views
- **Feature Selection (Phase 8):** interactive feature importance charts, side-by-side comparison of selection methods
- **Evaluation (Phase 10):** interactive confusion matrix, ROC/PR curves with threshold slider, residual plots, prediction inspection

### Stages that stay terminal-only
- Ingestion, cleaning, feature prep, modeling, tuning — decision-heavy surfaces where menus/prompts are faster than a web form
- Remote/SSH users are never forced through a browser for core workflow steps

### Implementation notes
- Streamlit apps read the same project state YAML and artifact files — no new storage layer
- Launched via `streamlit run` in a subprocess from the terminal; terminal resumes after user closes the view
- `streamlit` and `plotly` are optional dependencies, gated behind the same dependency manager used for `ydata-profiling`
- Plotext inline charts remain in V1 and V2 (fast inline feedback); Streamlit views are the optional deeper surface
