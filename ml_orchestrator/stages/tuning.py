import joblib
from datetime import datetime
from pathlib import Path

import pandas as pd
import questionary
from rich import box
from rich.console import Console
from rich.table import Table

from ml_orchestrator.deps.checker import check_and_install
from ml_orchestrator.stages.modeling import MODEL_REGISTRY
from ml_orchestrator.state.manager import save_state, update_state

console = Console()

REQUIRED_PACKAGES: list[str] = ["scikit-learn"]
OPTIONAL_PACKAGES: list[dict] = []


def run(state: dict) -> dict:
    console.print("\n[bold blue]── Hyperparameter Tuning ──[/bold blue]\n")

    if state.get("stages", {}).get("modeling") != "completed":
        console.print("[yellow]No baseline model found. Please run 'Train Baseline Model' first (option 8).[/yellow]")
        return state

    can_proceed, state = check_and_install(REQUIRED_PACKAGES, OPTIONAL_PACKAGES, state)
    if not can_proceed:
        return state

    X_train, X_test, y_train, y_test = _load_splits(state)
    if X_train is None:
        return state

    model_name = state["modeling"]["model_name"]
    problem_type = state.get("feature_preparation", {}).get("problem_type", "classification")
    model_cfg = _find_model_cfg(model_name, problem_type)
    if model_cfg is None:
        console.print(f"[red]Model '{model_name}' not found in registry. Cannot tune.[/red]")
        return state

    if not model_cfg["param_grid"]:
        console.print(f"[yellow]{model_name} has no param_grid defined — nothing to tune.[/yellow]")
        return state

    param_grid = _configure_param_grid(model_cfg)
    if param_grid is None:
        return state

    settings = state.get("settings", {})
    n_iter, cv_folds = _configure_search_settings(settings)
    if n_iter is None:
        return state

    best_params, cv_results = _run_search(
        model_cfg, param_grid, n_iter, cv_folds,
        X_train, y_train, problem_type, settings,
    )
    if best_params is None:
        return state

    original_params = state["modeling"].get("params", {})
    _show_comparison(original_params, best_params)

    retrain = questionary.confirm("Retrain with best parameters and save as tuned model?", default=True).ask()
    if retrain is None:
        return state

    if retrain:
        tuned_model, metrics = _retrain_and_eval(model_cfg, best_params, X_train, X_test, y_train, y_test, problem_type)
        if tuned_model is None:
            return state
        state = _save(tuned_model, model_name, best_params, original_params, metrics, cv_results, n_iter, cv_folds, state)
    else:
        console.print("[yellow]Tuning results recorded but model not retrained.[/yellow]\n")
        state = _save(None, model_name, best_params, original_params, {}, cv_results, n_iter, cv_folds, state)

    return state


# ---------------------------------------------------------------------------
# Load splits
# ---------------------------------------------------------------------------

def _load_splits(state: dict) -> tuple:
    project_dir = Path(state["project"]["project_dir"])
    artifacts = state.get("artifacts", {})

    x_train_key = "X_train_selected" if "X_train_selected" in artifacts else "X_train"
    x_test_key = "X_test_selected" if "X_test_selected" in artifacts else "X_test"

    try:
        X_train = pd.read_csv(project_dir / artifacts[x_train_key])
        X_test = pd.read_csv(project_dir / artifacts[x_test_key])
        y_train = pd.read_csv(project_dir / artifacts["y_train"]).squeeze()
        y_test = pd.read_csv(project_dir / artifacts["y_test"]).squeeze()
        return X_train, X_test, y_train, y_test
    except Exception as e:
        console.print(f"[red]Failed to load splits: {e}[/red]")
        return None, None, None, None


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------

def _find_model_cfg(model_name: str, problem_type: str) -> dict | None:
    for cfg in MODEL_REGISTRY.get(problem_type, []):
        if cfg["name"] == model_name:
            return cfg
    return None


# ---------------------------------------------------------------------------
# Param grid configuration
# ---------------------------------------------------------------------------

def _configure_param_grid(model_cfg: dict) -> dict | None:
    param_grid = dict(model_cfg["param_grid"])

    console.print("[bold]Parameter Grid[/bold]  [dim](values to search over)[/dim]\n")
    table = Table(box=box.SIMPLE, header_style="bold blue")
    table.add_column("Parameter", style="cyan")
    table.add_column("Values to try")
    for k, v in param_grid.items():
        table.add_row(k, str(v))
    console.print(table)

    accept = questionary.confirm("Use this parameter grid?", default=True).ask()
    if accept is None:
        return None
    if accept:
        return param_grid

    # Let user exclude params from search
    all_params = list(param_grid.keys())
    keep = questionary.checkbox(
        "Select parameters to include in search:",
        choices=all_params,
        default=all_params,
    ).ask()
    if keep is None:
        return None

    return {k: v for k, v in param_grid.items() if k in keep}


# ---------------------------------------------------------------------------
# Search settings
# ---------------------------------------------------------------------------

def _configure_search_settings(settings: dict) -> tuple[int | None, int | None]:
    n_iter = settings.get("tuning_n_iter", 20)
    cv_folds = settings.get("cv_folds", 5)

    console.print(
        f"\n  Search iterations: [cyan]{n_iter}[/cyan]  |  "
        f"CV folds: [cyan]{cv_folds}[/cyan]"
    )
    accept = questionary.confirm("Use these search settings?", default=True).ask()
    if accept is None:
        return None, None
    if accept:
        return n_iter, cv_folds

    n_iter_str = questionary.text(
        "Number of search iterations:",
        default=str(n_iter),
        validate=lambda v: v.isdigit() and int(v) > 0 or "Enter a positive integer.",
    ).ask()
    if n_iter_str is None:
        return None, None

    cv_str = questionary.text(
        "CV folds:",
        default=str(cv_folds),
        validate=lambda v: v.isdigit() and int(v) >= 2 or "Enter an integer >= 2.",
    ).ask()
    if cv_str is None:
        return None, None

    return int(n_iter_str), int(cv_str)


# ---------------------------------------------------------------------------
# Run RandomizedSearchCV
# ---------------------------------------------------------------------------

def _run_search(
    model_cfg: dict,
    param_grid: dict,
    n_iter: int,
    cv_folds: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    settings: dict,
) -> tuple[dict | None, dict]:
    import importlib
    from sklearn.model_selection import RandomizedSearchCV

    module_path, class_name = model_cfg["sklearn_class"].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
        base_model = ModelClass(random_state=settings.get("random_seed", 42))
    except Exception as e:
        console.print(f"[red]Failed to instantiate model: {e}[/red]")
        return None, {}

    scoring = "accuracy" if problem_type == "classification" else "r2"
    actual_iter = min(n_iter, _count_combinations(param_grid))

    console.print(
        f"\n[blue]Running RandomizedSearchCV "
        f"({actual_iter} iterations × {cv_folds} folds = {actual_iter * cv_folds} fits)...[/blue]"
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=actual_iter,
        cv=cv_folds,
        scoring=scoring,
        random_state=settings.get("random_seed", 42),
        n_jobs=-1,
        refit=False,
    )

    try:
        search.fit(X_train, y_train)
    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        return None, {}

    best_params = search.best_params_
    best_score = search.best_score_

    console.print(f"[green]Search complete. Best CV {scoring}: [bold]{best_score:.4f}[/bold][/green]\n")

    cv_results = {
        "best_score": round(float(best_score), 4),
        "scoring": scoring,
        "n_iter": actual_iter,
        "cv_folds": cv_folds,
    }
    return best_params, cv_results


def _count_combinations(param_grid: dict) -> int:
    total = 1
    for v in param_grid.values():
        total *= len(v)
    return total


# ---------------------------------------------------------------------------
# Show comparison
# ---------------------------------------------------------------------------

def _show_comparison(original: dict, best: dict) -> None:
    all_keys = sorted(set(original) | set(best))

    console.print("[bold]Parameter Comparison[/bold]\n")
    table = Table(box=box.SIMPLE, header_style="bold blue")
    table.add_column("Parameter", style="cyan")
    table.add_column("Original")
    table.add_column("Best found")
    table.add_column("")

    for key in all_keys:
        orig_val = str(original.get(key, "—"))
        best_val = str(best.get(key, "—"))
        changed = "[green]changed[/green]" if orig_val != best_val else "[dim]same[/dim]"
        table.add_row(key, orig_val, best_val, changed)

    console.print(table)


# ---------------------------------------------------------------------------
# Retrain and eval
# ---------------------------------------------------------------------------

def _retrain_and_eval(
    model_cfg: dict,
    best_params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    problem_type: str,
) -> tuple:
    import importlib

    module_path, class_name = model_cfg["sklearn_class"].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
        model = ModelClass(**best_params)
        model.fit(X_train, y_train)
    except Exception as e:
        console.print(f"[red]Retraining failed: {e}[/red]")
        return None, {}

    metrics: dict = {}
    try:
        if problem_type == "classification":
            from sklearn.metrics import accuracy_score
            train_acc = float(accuracy_score(y_train, model.predict(X_train)))
            test_acc = float(accuracy_score(y_test, model.predict(X_test)))
            metrics = {"train_accuracy": round(train_acc, 4), "test_accuracy": round(test_acc, 4)}
            console.print(f"  Train accuracy: [cyan]{train_acc:.4f}[/cyan]")
            console.print(f"  Test accuracy:  [cyan]{test_acc:.4f}[/cyan]\n")
        else:
            from sklearn.metrics import mean_absolute_error, r2_score
            test_r2 = float(r2_score(y_test, model.predict(X_test)))
            test_mae = float(mean_absolute_error(y_test, model.predict(X_test)))
            metrics = {"test_r2": round(test_r2, 4), "test_mae": round(test_mae, 4)}
            console.print(f"  Test R²:  [cyan]{test_r2:.4f}[/cyan]")
            console.print(f"  Test MAE: [cyan]{test_mae:.4f}[/cyan]\n")
    except Exception as e:
        console.print(f"[yellow]Could not compute metrics: {e}[/yellow]\n")

    return model, metrics


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save(
    model,
    model_name: str,
    best_params: dict,
    original_params: dict,
    metrics: dict,
    cv_results: dict,
    n_iter: int,
    cv_folds: int,
    state: dict,
) -> dict:
    project_dir = Path(state["project"]["project_dir"])

    if model is not None:
        tuned_path = project_dir / "artifacts" / "model_tuned.joblib"
        joblib.dump(model, tuned_path)
        console.print("[green]Tuned model saved to artifacts/model_tuned.joblib[/green]")
        if "artifacts" not in state:
            state["artifacts"] = {}
        state["artifacts"]["model_tuned"] = "artifacts/model_tuned.joblib"

    summary = {
        "completed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "model_name": model_name,
        "original_params": original_params,
        "best_params": best_params,
        "cv_results": cv_results,
        "retrained": model is not None,
        "tuned_metrics": metrics,
        "artifacts": {"model_tuned": "artifacts/model_tuned.joblib"} if model is not None else {},
    }

    state = update_state(state, tuning=summary)
    state["stages"]["tuning"] = "completed"
    save_state(state)

    console.print("[green]Hyperparameter tuning complete.[/green]\n")
    return state
