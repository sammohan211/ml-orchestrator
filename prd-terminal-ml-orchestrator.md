# Product Requirements Document (PRD)
## Project: ML-Orchestrator
## Version: 1.0
## Date: April 3, 2026
## Owner: Sam Mohan

---

## 1. Product Summary

The Terminal-Based Machine Learning Workflow Helper is a menu-driven terminal application designed to orchestrate common stages of tabular machine learning projects. The product guides users through data ingestion, profiling, cleaning, feature preparation, feature selection, baseline model training, evaluation, and artifact export.

The product aims to reduce repetitive boilerplate coding, standardize workflow structure, and preserve transparency by showing users what actions are being taken at each stage.

This project is intended as a practical ML productivity tool.

---

## 2. Problem

Machine learning workflows for tabular data are often fragmented across notebooks, scripts, and multiple independent Python packages. Users frequently spend significant time on repetitive setup tasks such as:

- loading and inspecting data
- checking schema and missing values
- cleaning and transforming columns
- selecting features
- fitting baseline models
- evaluating and exporting results
- maintaining reproducibility and consistency

This workflow is often manual and inconsistent, leading to fragile experiments that are difficult to share or rerun.

There is a need for a terminal-first product that unifies these steps into a guided workflow while keeping decisions visible and overridable.

---

## 3. Product Vision

Build a terminal-native ML workflow assistant that helps users move efficiently through the most common stages of tabular machine learning projects while maintaining transparency, reproducibility, and user control.

The product should feel like a structured workflow guide, not a black-box AutoML engine.

---

## 4. Goals

### Primary Goals
- reduce repetitive coding in early and mid-stage ML workflows
- guide users through a structured tabular ML project lifecycle
- provide a clear menu-driven experience in the terminal
- generate reproducible outputs such as configs, reports, and scripts
- show users what actions are being taken at each stage and why

### Secondary Goals
- support optional dependency installation when stages require missing packages
- make it easy to resume projects from saved state
- surface dataset-aware recommendations to guide cleaning and feature preparation decisions

---

## 5. Non-Goals

The following are explicitly out of scope for V1:

- multi-table datasets requiring joins or aggregations
- deep learning workflows
- image, video, audio, or NLP-specific pipelines
- model deployment or serving
- real-time monitoring
- distributed training
- web dashboard UI (Streamlit companion views for visual stages are a V2 candidate per §16 — not a dashboard replacement of the terminal workflow)
- feature store support
- end-to-end MLOps orchestration
- AutoML optimization across many model families
- multi-user collaboration
- cloud integration or remote execution

---

## 6. Target Users

The product is designed around two primary personas. Other groups benefit but are not the design center — when tradeoffs arise, decisions resolve in favor of the primary users.

### Primary Users
- **Solo ML practitioners and junior data scientists** working on tabular problems — have baseline ML knowledge, want to reduce boilerplate, value reproducibility and the ability to resume projects
- **ML engineers setting up baselines** — want a fast scaffold for new tabular problems before iterating or handing off to a deeper modeling workflow

### Secondary Users (benefit but do not drive design decisions)
- **Students learning applied ML** — the structured, transparent workflow serves as a learning scaffold, but the product is not an educational tool and does not over-explain
- **Data analysts with some ML exposure** — can produce baseline models without writing boilerplate, but are expected to understand concepts like imputation strategy, encoding, and train/test split

### Out of Frame
- Non-technical users expecting an AutoML black box — the product is a workflow guide, not an automated modeler
- Deep-learning, non-tabular, or production-serving workflows (see §5 Non-Goals)

### User Characteristics
- comfortable using a terminal, often in remote or SSH environments
- working primarily with tabular data
- have baseline ML knowledge — know what imputation, encoding, and train/test splits are
- want faster setup with less boilerplate, while keeping visibility into decisions
- prefer tools that don't hide what's happening

---

## 7. User Needs

Users need a tool that can:

- load and inspect datasets quickly
- surface data quality issues without custom notebook code
- guide them through cleaning and feature preparation
- run feature selection to reduce noise before modeling
- suggest sensible defaults without hiding implementation details
- run a baseline ML workflow quickly
- save outputs and decisions in reproducible formats
- start working without manually resolving dependency issues
- resume a project without losing prior decisions

---

## 8. Key Product Principles

### 8.1 Transparency
The product must show what it is doing, why it is doing it, and what artifacts are generated.

### 8.2 User Control
The product must allow users to accept, reject, skip, or override major workflow decisions.

### 8.3 Reproducibility
Workflow decisions, transformations, and outputs must be saved and audited so a project can be rerun or resumed later.

### 8.4 Modularity
Each workflow stage is self-contained and can be run, skipped, or revisited independently.

### 8.5 Terminal-First Experience
The terminal interface is the primary experience, not a temporary interface before a GUI. Browser-assisted visualization is a supported companion pattern — inline terminal charts are shown during the workflow, and richer reports (static HTML in V1; interactive Streamlit views as a V2 candidate per §16) may be opened in the browser at the user's discretion. Companion surfaces do not replace the terminal menu; the workflow backbone — navigation, prompts, decisions, logging, export — remains terminal-native.

---

## 9. Product Scope

### In Scope for V1

This is the single canonical scope list for V1. MVP and V1 are treated as equivalent — §15 references this list rather than duplicating it.

- terminal-based menu system
- new project creation
- existing project loading
- dependency checking and optional installation
- single flat-table CSV and Parquet dataset ingestion (no joins or multi-table workflows)
- schema inspection and profiling, including data quality warnings and leakage heuristics
- data cleaning — missing values, duplicates, column exclusion, outlier handling, and class imbalance handling
- feature preparation — target selection, train/test split, encoding, scaling
- feature selection with at least one method
- baseline classification and regression models with a pluggable model registry
- hyperparameter tuning (grid and randomized search)
- model evaluation with task-appropriate metrics
- inline terminal charts at profiling, feature selection, and evaluation stages
- static HTML reports opened in the browser at the user's option after key stages
- artifact export — cleaned dataset, config, feature list, evaluation summary, sklearn Pipeline, logs
- session persistence and resume

---

## 10. User Experience Overview

The user launches the application in a terminal and navigates a menu-based workflow.

A typical user path:

1. create or load a project
2. read a dataset
3. profile dataset and review data quality warnings
4. clean or exclude problematic columns
5. prepare features
6. select features
7. train a baseline model
8. optionally tune model parameters
9. evaluate results
10. export reports, config, and generated code

The interface should guide but not force a rigid order. Users should be able to revisit prior stages if needed.

After each major stage, the product displays a concise inline terminal chart summarizing key results (e.g. missing value distribution after profiling, feature importance after selection, metric summary after evaluation). The user is then prompted with the option to open a fuller HTML visual report in the browser.

---

## 11. Core User Stories

### Project Setup
- As a user, I want to create a project so I can organize artifacts and save workflow state.
- As a user, I want to load a saved project so I can continue where I left off.

### Data Ingestion
- As a user, I want to load a CSV or Parquet file so I can begin analysis quickly.
- As a user, I want to see row count, column count, and inferred types immediately after loading data.

### Profiling
- As a user, I want the system to summarize missing values, duplicates, and column characteristics so I can quickly understand dataset quality.
- As a user, I want the system to warn me about likely ID columns, constant columns, and possible leakage risks.

### Cleaning
- As a user, I want to choose how to handle missing values and duplicates so I can control downstream quality.
- As a user, I want to exclude columns from modeling so I can remove irrelevant or problematic features.
- As a user, I want cleaning decisions logged so the workflow is reproducible.

### Feature Preparation
- As a user, I want preprocessing recommendations based on column types so I can move faster without guessing.

### Feature Selection
- As a user, I want to run feature selection so I can reduce noise and focus the baseline model.
- As a user, I want to see which features were selected and which were removed so that I can understand what the model will use.

### Modeling
- As a user, I want the system to detect whether my task is classification or regression so setup is easier.
- As a user, I want to train a baseline model quickly so I can assess feasibility.
- As a user, I want to optionally tune basic model parameters so I can improve on the default baseline.

### Evaluation
- As a user, I want to see evaluation metrics appropriate to my task so I can judge model quality.
- As a user, I want a summary of model performance saved as an artifact so I can refer to it later.

### Export
- As a user, I want to export config, reports, and generated code so I can reuse the workflow outside the terminal application.
- As a user, I want to export the cleaned dataset so I can use it in other tools or pipeline

### Dependencies
- As a user, I want the system to detect missing packages and offer installation so I do not have to manually troubleshoot every stage.

---

## 12. Functional Requirements

### 12.1 Project Management
The product must allow users to:
- create a new project
- name the project
- choose a project directory
- save project metadata
- load an existing saved project
- persist workflow state after each major stage

Project state should include:
- dataset path
- project metadata
- inferred schema
- selected target column
- selected problem type
- cleaning actions
- selected features
- selected model and tuning parameters
- evaluation summaries
- dependency actions
- generated artifact paths

### 12.2 Terminal Navigation
The product must provide:
- a main menu
- stage-specific menus
- validated menu input
- clear navigation to move back, continue, or exit
- informative error messages for invalid selections

### 12.3 Data Ingestion
The product must support:
- CSV input
- Parquet input
- dataset preview
- interactive file path input with tab-completion, path validation, and clear errors for invalid, missing, or unsupported files
- row and column count
- inferred schema summary

**CSV Ingestion Robustness.** For CSV files, the product must:
- auto-detect the field delimiter (comma, tab, semicolon, pipe) with user override available
- auto-detect file encoding (UTF-8 with a fallback to latin-1) with user override available
- detect whether the first row is a header and allow the user to override the choice
- surface a structured error identifying the offending row and column when a file fails to parse, rather than a raw stack trace

### 12.4 Profiling
The product must provide:
- null count by column
- unique count by column
- duplicate row count
- basic statistics for numeric columns
- frequency summaries for categorical columns where feasible
- warnings for constant columns
- warnings for highly unique ID-like columns
- warnings for high-cardinality categorical features
- warnings for severe class imbalance where target exists
- warnings for outliers in numeric columns (z-score, IQR, or MAD-based detection), surfaced for use in the cleaning stage
- heuristic warnings for possible data leakage, covering at minimum:
  - column names matching suspicious patterns (e.g. names containing the target column name, or tokens such as `target`, `label`, `leak`)
  - features with near-perfect correlation to the target

Each leakage warning must:
- include the reasoning — which rule fired, which column, and the triggering value (e.g. correlation coefficient)
- be surfaced as a suspicion rather than a verdict; the product must not automatically drop or transform columns based on a leakage warning
- require explicit user action to accept (exclude from modeling) or dismiss, with the decision logged to project state

The product must display an inline terminal chart summarizing missing values and column distributions after profiling completes.

The product must prompt the user with the option to open a full HTML profiling report in the browser using the system default browser.

The product should support exporting a profiling artifact such as:
- HTML report
- JSON summary
- text summary

### 12.5 Cleaning
The product must allow users to:
- inspect missingness
- drop rows
- drop columns
- select imputation strategies (at minimum: mean, median, mode, constant value)
- remove duplicates
- exclude columns from downstream modeling
- review and save cleaning decisions

**Outlier Handling.** For each numeric column flagged as containing outliers during profiling, the product must:
- show outlier count and percentage per column
- allow the user to choose a detection method: z-score (default threshold ±3), IQR (1.5×IQR), or MAD
- offer treatment options per column: clip to boundary, remove rows, flag outliers as a new binary column, or skip
- log outlier decisions to project state

**Class Imbalance Handling.** For classification tasks where a target column has been set, the product must:
- show class distribution and imbalance ratio
- warn the user when the ratio exceeds a configurable threshold (default 5:1)
- offer treatment options: random oversampling of the minority class, random undersampling of the majority class, synthetic oversampling (SMOTE), use of class weights applied at modeling time, or skip
- apply sampling only to the training split, after the train/test split, to prevent leakage
- log imbalance strategy to project state
- degrade gracefully when optional dependencies (e.g. `imbalanced-learn`) are not installed — prompt the user via the dependency manager

### 12.6 Feature Preparation
The product must support:
- target column selection
- train/test split selection
- random seed entry
- optional stratification for classification
- recommendation of encoding strategy per categorical column
- recommendation of scaling strategy per numerical column
- user override of any recommendation on a per-column basis

**Encoding Options.** The product must support, at minimum:
- one-hot encoding
- ordinal encoding
- target encoding

**Scaling Options.** The product must support, at minimum:
- no scaling
- standard scaling (z-score)
- zero-centering (mean removal only)
- range scaling (min-max)
- robust scaling (for data with outliers)
- L2 normalization (per-sample unit norm)
- whitening (standard scaling combined with decorrelation)

**Algorithm-Applicability Hints.** When a user selects a scaler, the product must surface a hint if the choice is likely inappropriate for the selected model family — for example, warning when range scaling is applied before a tree-based model, since tree-based models are scale-invariant.

### 12.7 Feature Selection
The product must provide at least one of:
- variance threshold
- univariate selection
- model-based feature importance
- recursive feature elimination

The product must show:
- selected features
- removed features
- feature scores or ranking where available

The product must display an inline terminal chart of feature scores or importance after feature selection completes.

### 12.8 Baseline Modeling
The product must:
- infer likely problem type from the target column — classification when the target is categorical or a low-cardinality integer, regression otherwise
- allow user override of the inferred problem type
- provide a pluggable model registry so algorithms can be added without changes to stage code
- ship with the following baseline models in V1:
  - classification: Logistic Regression, Random Forest
  - regression: Linear Regression, Random Forest
- allow each model to declare its default parameters, tuning search space, supported metrics, and output artifacts
- allow the user to accept defaults or override basic parameters before training
- support hyperparameter search, gated by explicit user approval, with at minimum:
  - grid search with cross-validation
  - randomized search with cross-validation
- default to 5-fold cross-validation, user-configurable
- show best parameters and CV score improvement over the untuned baseline

### 12.9 Evaluation
The product must display metrics appropriate to the task.

Examples:
- classification: accuracy, precision, recall, F1
- regression: RMSE, MAE, R²

The product must generate a confusion matrix for classification tasks.

The product must save the evaluation summary as an artifact.

The product must generate a summary of model performance.

The product must display an inline terminal chart of key evaluation metrics after model evaluation completes.

The product must prompt the user with the option to open a full HTML evaluation report in the browser.

### 12.10 Artifact Export
The product must support exporting:
- cleaned dataset
- profiling report
- workflow config
- feature list
- model evaluation summary
- logs
- serialized sklearn Pipeline (preprocessing steps + model as a single exportable object)

### 12.11 Dependency Management
The product must:
- check dependencies before running a stage
- notify users when required packages are missing
- request approval before installation
- support installation in a controlled environment where possible
- log dependency actions
- recover gracefully from installation failure

### 12.12 Dataset-Aware Recommendations (cross-cutting)

The product must generate recommendations throughout the workflow that are informed by properties of the loaded dataset, not by fixed defaults alone. This operationalizes the §4 goal "surface dataset-aware recommendations to guide cleaning and feature preparation decisions."

**Signals used.** Recommendations must draw on at minimum:
- column dtype (numeric, categorical, datetime, string)
- cardinality and uniqueness
- missingness rate and pattern
- skewness and presence of outliers for numeric columns
- correlation with the target (where target is known)
- selected problem type and model family

**Stages that produce dataset-aware recommendations.**
- profiling — quality warnings, leakage heuristics, outlier detection, class imbalance detection
- cleaning — imputation strategy suggestions per column, outlier treatment defaults
- feature preparation — encoding strategy per categorical column, scaling strategy per numeric column, algorithm-applicability hints
- feature selection — method-appropriate defaults for threshold, k, or estimator choice based on dataset size and target type

**Interaction model.** Every recommendation must:
- be shown to the user alongside the signal that produced it
- allow the user to accept, override, or skip
- never be auto-applied without a confirmation step
- have its outcome (accepted, overridden, skipped) logged to project state for reproducibility

---

## 13. Non-Functional Requirements

### Usability
- menus must be readable and easy to navigate
- prompts must be clear and descriptive
- stage output must be understandable to a user with basic ML concepts but not an expert

### Reliability
- invalid file paths must be handled gracefully
- unsupported file formats must return a clear error
- keyboard interrupts and unexpected exits must not corrupt saved project state

### Performance
- terminal interaction should remain responsive on moderate-size tabular datasets - datasets upto ~ 1M rows and ~500 columns on a standard laptop
- long-running stages should show visible progress or status updates

### Maintainability
- stage logic must be modular and separable
- each workflow stage must be independently testable and replaceable
- the codebase should support future extension without a major rewrite

### Portability
- the application should run on Linux, macOS, and Windows where supported by Python dependencies

### Reproducibility
- workflow state and decisions must be serializable and reloadable
- reloaded state must produce the same results when rerun on the same dataset

### Transparency
- actions performed by the product must be visible in terminal output and/or logs

### Security
- dependency installation must require explicit user approval and must not execute arbitrary code

---

## 14. Acceptance Criteria and Success Indicators

### 14.1 Functional Acceptance Criteria

The product will be considered functionally complete for V1 if a user can:

- create a project from the terminal
- load a CSV or Parquet dataset
- review a profiling summary
- perform cleaning actions — handle missing values, remove duplicates, exclude unwanted columns, treat outliers, and address class imbalance where present
- run feature preparation and selection
- train a baseline model
- view evaluation metrics
- optionally tune model parameters and see the impact on evaluation metrics
- export artifacts and workflow configuration
- resume the project later without losing state

### 14.2 Quantitative Performance Targets

| Metric | Target |
|--------|--------|
| Time from project creation to first trained baseline model | < 10 minutes on a tabular dataset ≤ 100k rows and ≤ 50 columns |
| Full end-to-end workflow completion time (ingest → export) | < 20 minutes on a tabular dataset ≤ 100k rows and ≤ 50 columns |
| Maximum supported dataset size (interactive responsiveness) | ~1M rows, ~500 columns on a standard laptop (see §13 Performance) |
| Unhandled errors during a standard workflow run on a valid dataset | 0 |
| Reproducibility: exported config rerun on the same dataset | produces identical outputs |
| Reproducibility: exported sklearn Pipeline on the same dataset | produces identical predictions |
| State recovery after reload | all prior decisions and artifacts restored without loss |

### 14.3 Qualitative Success Indicators

Observed rather than strictly measured:

- users spend less time on setup boilerplate than they would writing equivalent notebook code from scratch
- users can explain what the application did at each stage
- generated artifacts are sufficient for reruns, review, and handoff
- users trust the workflow enough to run it on non-toy datasets after initial learning

---

## 15. MVP Definition

MVP and V1 are equivalent. The product ships as a single release — there is no phased MVP-then-V1 distinction.

- **Scope included in the MVP:** as defined in §9 "In Scope for V1"
- **Scope excluded from the MVP:** as defined in §5 "Non-Goals"

This section exists to make the equivalence explicit and to prevent the MVP and V1 scope lists from drifting apart over time.

---

## 16. Risks and Open Questions

### Risks
- over-automation may reduce user trust
- dependency installation may behave differently across environments
- inline charts and browser-assisted HTML reports mitigate visualization limits, but terminal chart fidelity remains constrained
- scope creep may weaken overall polish
- leakage detection heuristics may produce false positives — mitigated by surfacing each warning with its reasoning (rule, column, triggering value), never auto-applying action, and requiring explicit user accept or dismiss (see §12.4)
- state corruption on interrupted runs
- imputation and encoding choices silently affecting results

### Open Questions
- should package installs happen in a project-specific virtual environment by default? 
Yes, strongly recommended 
- should the product generate a Python script, notebook, or both in V1? Scripts for V1, no notebooks
- should profiling use a built-in summary first and richer external reporting optionally? Built-in profiling
- should the application support a command mode alongside the interactive menu mode? Menu for V1
- should a future version offer a Streamlit-based UI as an alternative to the terminal interface? V2 direction resolved: Streamlit + Plotly as *companion* surfaces for visual, exploratory stages only — profiling, feature selection, evaluation. Launched on-demand from the terminal, replacing the static HTML reports described in §12.4 and §12.9. The terminal remains the workflow backbone (navigation, prompts, decisions, logging, export). Ingestion, cleaning, feature prep, modeling, and tuning stay terminal-only — decision-heavy surfaces where prompts are faster than a web form, and remote/SSH users are never forced through a browser. Ingestion in V1 uses tab-completing path input to reduce terminal jankiness without requiring Streamlit.

---

## 17. Recommended Technical Approach

### Suggested Stack
- Python
- `rich` for terminal output formatting and interactive menus (primary UI driver)
- `questionary` for user prompts and menu input
- `plotext` for inline terminal charts (distributions, feature importance, metrics)
- `typer` or `click` for CLI entry point and optional command mode in future versions
- `textual` as an optional upgrade path for a full TUI layout if needed
- `pandas` for data handling (polars is a future alternative if performance becomes a concern)
- `scikit-learn` for preprocessing, feature selection, and baseline models
- `pyyaml` for project config and state (human-readable and inspectable)
- `joblib` for model serialization
- `venv` and `subprocess` for environment and package handling
- optional profiling support via `ydata-profiling`
- **V2 companion UI** (post-V1): `streamlit` + `plotly` for interactive visual surfaces at profiling, feature selection, and evaluation stages only. Launched on-demand from the terminal; replaces the static HTML report prompts from V1. Does not replace the terminal menu. Ingestion, cleaning, feature prep, modeling, and tuning remain terminal-only.

### Architectural Components
- terminal UI layer
- project/session state manager
- config/defaults layer (sensible per-stage defaults, separate from state)
- stage engine
- dependency manager
- artifact manager
- ML workflow modules

---

## 18. Example Main Menu

```text
ML Workflow Helper
------------------
1. Start New Project
2. Load Existing Project
3. Read Dataset
4. Profile Dataset
5. Clean Dataset
6. Prepare Features
7. Select Features
8. Train Baseline Model
9. Tune Hyperparameters
10. Evaluate Model
11. Export Artifacts
12. Settings
13. Exit
