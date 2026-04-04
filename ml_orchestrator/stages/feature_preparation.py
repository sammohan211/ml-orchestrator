import joblib
from datetime import datetime
from pathlib import Path

import pandas as pd
import questionary
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ml_orchestrator.deps.checker import check_and_install
from ml_orchestrator.state.manager import save_state, update_state

console = Console()

REQUIRED_PACKAGES: list[str] = ["scikit-learn"]
OPTIONAL_PACKAGES: list[dict] = []


def run(state: dict) -> dict:
    console.print("\n[bold blue]── Prepare Features ──[/bold blue]\n")

    if state.get("stages", {}).get("ingestion") != "completed":
        console.print("[yellow]No dataset loaded. Please run 'Read Dataset' first (option 3).[/yellow]")
        return state

    can_proceed, state = check_and_install(REQUIRED_PACKAGES, OPTIONAL_PACKAGES, state)
    if not can_proceed:
        return state

    df = _load_dataset(state)
    if df is None:
        return state

    df = _drop_excluded(df, state)

    target_col = _select_target(df, state)
    if target_col is None:
        return state

    problem_type, problem_type_source = _confirm_problem_type(df[target_col], state)
    if problem_type is None:
        return state

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = _split(X, y, problem_type, state)
    if X_train is None:
        return state

    encoding_plan = _plan_encoding(X_train, state)
    scaling_plan = _plan_scaling(X_train)

    encoding_plan = _show_and_override_encoding(encoding_plan, X_train)
    if encoding_plan is None:
        return state
    scaling_plan = _show_and_override_scaling(scaling_plan, X_train)
    if scaling_plan is None:
        return state

    preprocessor, X_train_df, X_test_df = _apply_transforms(
        X_train, X_test, encoding_plan, scaling_plan
    )
    if preprocessor is None:
        return state

    state = _save(
        X_train_df, X_test_df, y_train, y_test,
        preprocessor,
        target_col, problem_type, problem_type_source,
        encoding_plan, scaling_plan,
        X_train, X_test,
        state,
    )
    return state


# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------

def _load_dataset(state: dict) -> pd.DataFrame | None:
    # Prefer cleaned dataset if available
    cleaned = state.get("artifacts", {}).get("cleaned_dataset")
    project_dir = Path(state["project"]["project_dir"])

    if cleaned:
        path = project_dir / cleaned
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as e:
                console.print(f"[yellow]Could not load cleaned dataset ({e}), falling back to source.[/yellow]")

    source_path = state.get("ingestion", {}).get("source_path")
    if not source_path:
        console.print("[red]No dataset path found in state.[/red]")
        return None

    path = Path(source_path)
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        encoding = state["ingestion"].get("encoding", "utf-8")
        delimiter = state["ingestion"].get("delimiter", ",")
        return pd.read_csv(path, encoding=encoding, sep=delimiter)
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        return None


def _drop_excluded(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    excluded = state.get("cleaning", {}).get("excluded_columns", [])
    to_drop = [c for c in excluded if c in df.columns]
    if to_drop:
        console.print(f"[dim]Excluding columns from cleaning stage: {', '.join(to_drop)}[/dim]\n")
        df = df.drop(columns=to_drop)

    # Apply data dictionary renames
    dd = state.get("data_dictionary", {})
    rename_map = {
        col: meta["renamed_to"]
        for col, meta in dd.get("columns", {}).items()
        if meta and meta.get("renamed_to") and col in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def _select_target(df: pd.DataFrame, state: dict) -> str | None:
    existing = state.get("feature_preparation", {}).get("target_column")
    console.print(Panel(
        "Select the column you want to predict (the target / label).",
        title="Target Column",
        border_style="cyan",
    ))

    default = existing if existing and existing in df.columns else None
    choices = list(df.columns)

    target = questionary.select(
        "Target column:",
        choices=choices,
        default=default,
    ).ask()
    return target


# ---------------------------------------------------------------------------
# Problem type
# ---------------------------------------------------------------------------

def _infer_problem_type(series: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    if series.nunique() <= 20:
        return "classification"
    return "regression"


def _confirm_problem_type(series: pd.Series, state: dict) -> tuple[str | None, str]:
    inferred = _infer_problem_type(series)
    existing = state.get("feature_preparation", {}).get("problem_type")
    default = existing or inferred

    console.print(f"\n  Inferred problem type: [cyan]{inferred}[/cyan]  "
                  f"(target dtype={series.dtype}, {series.nunique()} unique values)")

    problem_type = questionary.select(
        "Problem type:",
        choices=["classification", "regression"],
        default=default,
    ).ask()
    if problem_type is None:
        return None, "inferred"

    source = "inferred" if problem_type == inferred else "user_override"
    return problem_type, source


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def _split(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    state: dict,
) -> tuple:
    from sklearn.model_selection import train_test_split

    settings = state.get("settings", {})
    test_size = settings.get("test_size", 0.2)
    random_seed = settings.get("random_seed", 42)
    stratified = settings.get("stratified", True) and problem_type == "classification"

    console.print(
        f"\n  Split: [cyan]{int((1 - test_size) * 100)}% train / {int(test_size * 100)}% test[/cyan]  |  "
        f"seed=[cyan]{random_seed}[/cyan]  |  "
        f"stratified=[cyan]{stratified}[/cyan]"
    )
    confirm = questionary.confirm("Use these split settings?", default=True).ask()
    if confirm is None:
        return None, None, None, None

    if not confirm:
        test_size_str = questionary.text(
            "Test size (0.0–0.5):",
            default=str(test_size),
            validate=lambda v: _valid_float(v, 0.0, 0.5),
        ).ask()
        if test_size_str is None:
            return None, None, None, None
        test_size = float(test_size_str)

        seed_str = questionary.text("Random seed:", default=str(random_seed)).ask()
        if seed_str is None:
            return None, None, None, None
        random_seed = int(seed_str)

    stratify = y if stratified else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=stratify
    )
    console.print(
        f"  [green]Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows[/green]\n"
    )
    return X_train, X_test, y_train, y_test


def _valid_float(v: str, lo: float, hi: float) -> bool | str:
    try:
        f = float(v)
        if lo <= f <= hi:
            return True
        return f"Enter a value between {lo} and {hi}."
    except ValueError:
        return "Enter a number."


# ---------------------------------------------------------------------------
# Encoding plan
# ---------------------------------------------------------------------------

def _plan_encoding(X: pd.DataFrame, state: dict) -> dict[str, dict]:
    settings = state.get("settings", {})
    high_card = settings.get("high_cardinality_threshold", 50)

    plan: dict[str, dict] = {}
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    for col in cat_cols:
        n = X[col].nunique()
        if n == 2:
            strategy, reason = "ordinal", "binary column — 2 unique values"
        elif n <= high_card:
            strategy, reason = "one_hot", f"{n} unique values (≤ threshold {high_card})"
        else:
            strategy, reason = "ordinal", f"high cardinality — {n} unique values (> threshold {high_card})"
        plan[col] = {"strategy": strategy, "source": "recommended", "reason": reason, "n_unique": n}

    return plan


def _show_and_override_encoding(plan: dict, X: pd.DataFrame) -> dict | None:
    if not plan:
        console.print("[dim]No categorical columns — encoding step skipped.[/dim]\n")
        return plan

    console.print("[bold]Encoding Recommendations[/bold]\n")
    table = Table(box=box.SIMPLE, header_style="bold blue")
    table.add_column("Column", style="cyan")
    table.add_column("Unique", justify="right")
    table.add_column("Strategy", justify="center")
    table.add_column("Reason", style="dim")
    for col, meta in plan.items():
        table.add_row(col, str(meta["n_unique"]), meta["strategy"], meta["reason"])
    console.print(table)

    while True:
        override = questionary.confirm("Override any encoding strategy?", default=False).ask()
        if override is None:
            return None
        if not override:
            break
        col = questionary.select("Column to override:", choices=list(plan.keys())).ask()
        if col is None:
            return None
        new_strategy = questionary.select(
            f"Strategy for '{col}':",
            choices=["one_hot", "ordinal"],
            default=plan[col]["strategy"],
        ).ask()
        if new_strategy is None:
            return None
        plan[col]["strategy"] = new_strategy
        plan[col]["source"] = "user_override"
        console.print(f"  [green]Updated '{col}' → {new_strategy}[/green]")

    return plan


# ---------------------------------------------------------------------------
# Scaling plan
# ---------------------------------------------------------------------------

def _plan_scaling(X: pd.DataFrame) -> dict[str, dict]:
    plan: dict[str, dict] = {}
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    for col in num_cols:
        plan[col] = {"strategy": "standard", "source": "recommended"}
    return plan


def _show_and_override_scaling(plan: dict, X: pd.DataFrame) -> dict | None:
    if not plan:
        console.print("[dim]No numeric columns — scaling step skipped.[/dim]\n")
        return plan

    console.print("[bold]Scaling Recommendations[/bold]\n")
    table = Table(box=box.SIMPLE, header_style="bold blue")
    table.add_column("Column", style="cyan")
    table.add_column("Strategy", justify="center")
    table.add_column("Source", style="dim")
    for col, meta in plan.items():
        table.add_row(col, meta["strategy"], meta["source"])
    console.print(table)

    while True:
        override = questionary.confirm("Override any scaling strategy?", default=False).ask()
        if override is None:
            return None
        if not override:
            break
        col = questionary.select("Column to override:", choices=list(plan.keys())).ask()
        if col is None:
            return None
        new_strategy = questionary.select(
            f"Strategy for '{col}':",
            choices=["standard", "minmax", "none"],
            default=plan[col]["strategy"],
        ).ask()
        if new_strategy is None:
            return None
        plan[col]["strategy"] = new_strategy
        plan[col]["source"] = "user_override"
        console.print(f"  [green]Updated '{col}' → {new_strategy}[/green]")

    return plan


# ---------------------------------------------------------------------------
# Apply transforms
# ---------------------------------------------------------------------------

def _apply_transforms(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    encoding_plan: dict,
    scaling_plan: dict,
) -> tuple:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

    transformers = []

    one_hot_cols = [c for c, m in encoding_plan.items() if m["strategy"] == "one_hot" and c in X_train.columns]
    ordinal_cols = [c for c, m in encoding_plan.items() if m["strategy"] == "ordinal" and c in X_train.columns]
    standard_cols = [c for c, m in scaling_plan.items() if m["strategy"] == "standard" and c in X_train.columns]
    minmax_cols = [c for c, m in scaling_plan.items() if m["strategy"] == "minmax" and c in X_train.columns]
    if one_hot_cols:
        transformers.append(("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), one_hot_cols))
    if ordinal_cols:
        transformers.append(("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ordinal_cols))
    if standard_cols:
        transformers.append(("standard", StandardScaler(), standard_cols))
    if minmax_cols:
        transformers.append(("minmax", MinMaxScaler(), minmax_cols))

    # Remaining cols (no_scale_cols + any unaccounted) pass through
    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")

    try:
        X_train_arr = preprocessor.fit_transform(X_train)
        X_test_arr = preprocessor.transform(X_test)
    except Exception as e:
        console.print(f"[red]Failed to apply transforms: {e}[/red]")
        return None, None, None

    feature_names = _clean_feature_names(preprocessor.get_feature_names_out())
    X_train_df = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test.index)

    console.print(f"\n  [green]Transformed: {len(feature_names)} features after encoding/scaling.[/green]\n")
    return preprocessor, X_train_df, X_test_df


def _clean_feature_names(raw_names) -> list[str]:
    """Strip sklearn ColumnTransformer prefixes (e.g. 'one_hot__gender_Female' → 'gender_Female')."""
    cleaned = []
    for name in raw_names:
        parts = str(name).split("__", 1)
        cleaned.append(parts[1] if len(parts) > 1 else name)
    return cleaned


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor,
    target_col: str,
    problem_type: str,
    problem_type_source: str,
    encoding_plan: dict,
    scaling_plan: dict,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    state: dict,
) -> dict:
    settings = state.get("settings", {})
    test_size = settings.get("test_size", 0.2)
    random_seed = settings.get("random_seed", 42)
    stratified = settings.get("stratified", True) and problem_type == "classification"

    project_dir = Path(state["project"]["project_dir"])
    artifacts_dir = project_dir / "artifacts"

    X_train.to_csv(artifacts_dir / "X_train.csv", index=False)
    X_test.to_csv(artifacts_dir / "X_test.csv", index=False)
    y_train.to_csv(artifacts_dir / "y_train.csv", index=False)
    y_test.to_csv(artifacts_dir / "y_test.csv", index=False)

    joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")

    feature_names = list(X_train.columns)
    (artifacts_dir / "features.txt").write_text("\n".join(feature_names), encoding="utf-8")

    console.print("[green]Artifacts saved:[/green]")
    for name in ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "preprocessor.joblib", "features.txt"):
        console.print(f"  artifacts/{name}")

    # Strip internal keys before saving to state
    enc_state = {col: {"strategy": m["strategy"], "source": m["source"]} for col, m in encoding_plan.items()}
    scl_state = {col: {"strategy": m["strategy"], "source": m["source"]} for col, m in scaling_plan.items()}

    summary = {
        "completed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "target_column": target_col,
        "problem_type": problem_type,
        "problem_type_source": problem_type_source,
        "split": {
            "test_size": test_size,
            "random_seed": random_seed,
            "stratified": stratified,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
        "encoding": enc_state,
        "scaling": scl_state,
        "feature_count": len(feature_names),
        "artifacts": {
            "X_train": "artifacts/X_train.csv",
            "X_test": "artifacts/X_test.csv",
            "y_train": "artifacts/y_train.csv",
            "y_test": "artifacts/y_test.csv",
            "preprocessor": "artifacts/preprocessor.joblib",
            "feature_list": "artifacts/features.txt",
        },
    }

    if "artifacts" not in state:
        state["artifacts"] = {}
    state["artifacts"].update({
        "X_train": "artifacts/X_train.csv",
        "X_test": "artifacts/X_test.csv",
        "y_train": "artifacts/y_train.csv",
        "y_test": "artifacts/y_test.csv",
        "preprocessor": "artifacts/preprocessor.joblib",
        "feature_list": "artifacts/features.txt",
    })

    state = update_state(state, feature_preparation=summary)
    state["stages"]["feature_preparation"] = "completed"
    save_state(state)

    console.print("\n[green]Feature preparation complete.[/green]\n")
    return state
