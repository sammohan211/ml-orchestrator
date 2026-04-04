from datetime import datetime
from pathlib import Path

import pandas as pd
import questionary
from rich import box
from rich.console import Console
from rich.table import Table

from ml_orchestrator.deps.checker import check_and_install
from ml_orchestrator.state.manager import save_state, update_state
from ml_orchestrator.ui import charts

console = Console()

REQUIRED_PACKAGES: list[str] = ["scikit-learn"]
OPTIONAL_PACKAGES: list[dict] = []

METHODS = {
    "Model-based importance (Random Forest)": "model_based",
    "Correlation threshold": "correlation",
}


def run(state: dict) -> dict:
    console.print("\n[bold blue]── Select Features ──[/bold blue]\n")

    if state.get("stages", {}).get("feature_preparation") != "completed":
        console.print("[yellow]Features not prepared. Please run 'Prepare Features' first (option 6).[/yellow]")
        return state

    can_proceed, state = check_and_install(REQUIRED_PACKAGES, OPTIONAL_PACKAGES, state)
    if not can_proceed:
        return state

    X_train, X_test, y_train = _load_splits(state)
    if X_train is None:
        return state

    problem_type = state.get("feature_preparation", {}).get("problem_type", "classification")
    settings = state.get("settings", {})
    threshold = settings.get("feature_selection_threshold", 0.01)

    method_label = questionary.select(
        "Feature selection method:",
        choices=list(METHODS.keys()),
    ).ask()
    if method_label is None:
        return state

    method = METHODS[method_label]

    if method == "model_based":
        scores, method_params = _model_based_scores(X_train, y_train, problem_type, settings)
    else:
        scores, method_params = _correlation_scores(X_train, y_train)

    if scores is None:
        return state

    selected, removed, threshold = _select_with_threshold(scores, threshold, method)
    if selected is None:
        return state

    state = _save(X_train, X_test, scores, selected, removed, threshold, method, method_params, state)
    return state


# ---------------------------------------------------------------------------
# Load splits
# ---------------------------------------------------------------------------

def _load_splits(state: dict) -> tuple:
    project_dir = Path(state["project"]["project_dir"])
    artifacts = state.get("artifacts", {})

    try:
        X_train = pd.read_csv(project_dir / artifacts["X_train"])
        X_test = pd.read_csv(project_dir / artifacts["X_test"])
        y_train = pd.read_csv(project_dir / artifacts["y_train"]).squeeze()
        return X_train, X_test, y_train
    except Exception as e:
        console.print(f"[red]Failed to load prepared splits: {e}[/red]")
        return None, None, None


# ---------------------------------------------------------------------------
# Scoring methods
# ---------------------------------------------------------------------------

def _model_based_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    settings: dict,
) -> tuple[dict | None, dict]:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    n_estimators = 50
    random_state = settings.get("random_seed", 42)

    if problem_type == "classification":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    console.print("[blue]Fitting Random Forest to compute feature importances...[/blue]")
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        console.print(f"[red]Failed to fit model: {e}[/red]")
        return None, {}

    scores = dict(zip(X_train.columns, model.feature_importances_))
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    method_params = {
        "estimator": type(model).__name__,
        "n_estimators": n_estimators,
        "random_state": random_state,
    }
    return scores, method_params


def _correlation_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict | None, dict]:
    # Score = absolute correlation with target (for numeric target/features)
    # For classification targets, use point-biserial or just encode as int
    try:
        y_numeric = pd.to_numeric(y_train, errors="coerce")
        if y_numeric.isnull().all():
            # Encode categorical target as codes
            y_numeric = y_train.astype("category").cat.codes.astype(float)

        num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
        if not num_cols:
            console.print("[red]Correlation method requires numeric features. Use model-based instead.[/red]")
            return None, {}

        scores = {}
        for col in X_train.columns:
            if col in num_cols:
                scores[col] = abs(float(X_train[col].corr(y_numeric)))
            else:
                scores[col] = 0.0

        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        method_params = {"metric": "absolute_correlation_with_target"}
        return scores, method_params

    except Exception as e:
        console.print(f"[red]Failed to compute correlations: {e}[/red]")
        return None, {}


# ---------------------------------------------------------------------------
# Threshold selection UI
# ---------------------------------------------------------------------------

def _select_with_threshold(
    scores: dict,
    threshold: float,
    method: str,
) -> tuple[list | None, list | None, float]:
    score_label = "Importance" if method == "model_based" else "Correlation"

    while True:
        selected = [f for f, s in scores.items() if s >= threshold]
        removed = [f for f, s in scores.items() if s < threshold]

        _show_scores_table(scores, threshold, score_label)
        _show_chart(scores, score_label)

        console.print(
            f"\n  Threshold: [cyan]{threshold}[/cyan]  |  "
            f"Selected: [green]{len(selected)}[/green]  |  "
            f"Removed: [red]{len(removed)}[/red]\n"
        )

        if not selected:
            console.print("[red]No features selected at this threshold. Lower it.[/red]")
            new_val = questionary.text(
                "New threshold:",
                validate=lambda v: _valid_threshold(v),
            ).ask()
            if new_val is None:
                return None, None, threshold
            threshold = float(new_val)
            continue

        action = questionary.select(
            "What would you like to do?",
            choices=[
                "Accept and continue",
                "Adjust threshold",
            ],
        ).ask()
        if action is None:
            return None, None, threshold
        if action == "Accept and continue":
            return selected, removed, threshold

        new_val = questionary.text(
            "New threshold:",
            default=str(threshold),
            validate=lambda v: _valid_threshold(v),
        ).ask()
        if new_val is None:
            return None, None, threshold
        threshold = float(new_val)


def _valid_threshold(v: str) -> bool | str:
    try:
        f = float(v)
        if 0.0 <= f <= 1.0:
            return True
        return "Enter a value between 0.0 and 1.0."
    except ValueError:
        return "Enter a number."


def _show_scores_table(scores: dict, threshold: float, score_label: str) -> None:
    console.print(f"\n[bold]Feature Scores[/bold]  [dim](threshold = {threshold})[/dim]\n")
    table = Table(box=box.SIMPLE, header_style="bold blue")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Feature", style="cyan")
    table.add_column(score_label, justify="right")
    table.add_column("", justify="center")

    for i, (feature, score) in enumerate(scores.items(), 1):
        status = "[green]✓[/green]" if score >= threshold else "[red]✗[/red]"
        score_str = f"{score:.4f}"
        if score < threshold:
            score_str = f"[dim]{score_str}[/dim]"
        table.add_row(str(i), feature, score_str, status)

    console.print(table)


def _show_chart(scores: dict, score_label: str) -> None:
    top = list(scores.items())[:20]
    if not top:
        return
    labels = [f for f, _ in top]
    values = [round(s, 4) for _, s in top]
    charts.horizontal_bar(labels, values, title=f"Feature {score_label} (top {len(top)})", unit=score_label.lower())


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scores: dict,
    selected: list,
    removed: list,
    threshold: float,
    method: str,
    method_params: dict,
    state: dict,
) -> dict:
    project_dir = Path(state["project"]["project_dir"])
    artifacts_dir = project_dir / "artifacts"

    X_train_sel = X_train[selected]
    X_test_sel = X_test[selected]

    X_train_sel.to_csv(artifacts_dir / "X_train_selected.csv", index=False)
    X_test_sel.to_csv(artifacts_dir / "X_test_selected.csv", index=False)
    (artifacts_dir / "features_selected.txt").write_text("\n".join(selected), encoding="utf-8")

    console.print("\n[green]Artifacts saved:[/green]")
    for name in ("X_train_selected.csv", "X_test_selected.csv", "features_selected.txt"):
        console.print(f"  artifacts/{name}")

    summary = {
        "completed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "method": method,
        "method_params": {**method_params, "threshold": threshold},
        "selected_features": selected,
        "removed_features": removed,
        "feature_scores": {f: round(s, 6) for f, s in scores.items()},
        "artifacts": {
            "X_train_selected": "artifacts/X_train_selected.csv",
            "X_test_selected": "artifacts/X_test_selected.csv",
            "feature_list_selected": "artifacts/features_selected.txt",
        },
    }

    if "artifacts" not in state:
        state["artifacts"] = {}
    state["artifacts"].update({
        "X_train_selected": "artifacts/X_train_selected.csv",
        "X_test_selected": "artifacts/X_test_selected.csv",
        "feature_list_selected": "artifacts/features_selected.txt",
    })

    state = update_state(state, feature_selection=summary)
    state["stages"]["feature_selection"] = "completed"
    save_state(state)

    console.print(
        f"\n[green]Feature selection complete. "
        f"{len(selected)} features selected, {len(removed)} removed.[/green]\n"
    )
    return state
