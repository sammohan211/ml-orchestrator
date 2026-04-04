import importlib
import joblib
from datetime import datetime
from pathlib import Path

import pandas as pd
import questionary
from rich import box
from rich.console import Console
from rich.table import Table

from ml_orchestrator.deps.checker import check_and_install
from ml_orchestrator.state.manager import save_state, update_state

console = Console()

REQUIRED_PACKAGES: list[str] = ["scikit-learn"]
OPTIONAL_PACKAGES: list[dict] = []

# ---------------------------------------------------------------------------
# Model registry
# Add new models here — no other code changes needed.
# Each entry: name, sklearn_class (dotted path), default_params, param_grid
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, list[dict]] = {
    "classification": [
        {
            "name": "RandomForestClassifier",
            "sklearn_class": "sklearn.ensemble.RandomForestClassifier",
            "default_params": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "random_state": 42},
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        {
            "name": "LogisticRegression",
            "sklearn_class": "sklearn.linear_model.LogisticRegression",
            "default_params": {"C": 1.0, "max_iter": 1000, "random_state": 42},
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["lbfgs", "liblinear"],
            },
        },
        {
            "name": "GradientBoostingClassifier",
            "sklearn_class": "sklearn.ensemble.GradientBoostingClassifier",
            "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10],
            },
        },
    ],
    "regression": [
        {
            "name": "RandomForestRegressor",
            "sklearn_class": "sklearn.ensemble.RandomForestRegressor",
            "default_params": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "random_state": 42},
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
        },
        {
            "name": "LinearRegression",
            "sklearn_class": "sklearn.linear_model.LinearRegression",
            "default_params": {},
            "param_grid": {},
        },
        {
            "name": "GradientBoostingRegressor",
            "sklearn_class": "sklearn.ensemble.GradientBoostingRegressor",
            "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
            },
        },
    ],
}


def run(state: dict) -> dict:
    console.print("\n[bold blue]── Train Baseline Model ──[/bold blue]\n")

    if state.get("stages", {}).get("feature_preparation") != "completed":
        console.print("[yellow]Features not prepared. Please run 'Prepare Features' first (option 6).[/yellow]")
        return state

    can_proceed, state = check_and_install(REQUIRED_PACKAGES, OPTIONAL_PACKAGES, state)
    if not can_proceed:
        return state

    X_train, X_test, y_train, y_test = _load_splits(state)
    if X_train is None:
        return state

    problem_type = state.get("feature_preparation", {}).get("problem_type", "classification")
    available = MODEL_REGISTRY.get(problem_type, [])
    if not available:
        console.print(f"[red]No models registered for problem type '{problem_type}'.[/red]")
        return state

    model_cfg = _select_model(available, state)
    if model_cfg is None:
        return state

    params = _configure_params(model_cfg)
    if params is None:
        return state

    model = _train(model_cfg, params, X_train, y_train)
    if model is None:
        return state

    metrics = _quick_eval(model, X_train, X_test, y_train, y_test, problem_type)

    state = _save(model, model_cfg, params, metrics, state)
    return state


# ---------------------------------------------------------------------------
# Load splits (prefer selected features if available)
# ---------------------------------------------------------------------------

def _load_splits(state: dict) -> tuple:
    project_dir = Path(state["project"]["project_dir"])
    artifacts = state.get("artifacts", {})

    x_train_key = "X_train_selected" if "X_train_selected" in artifacts else "X_train"
    x_test_key = "X_test_selected" if "X_test_selected" in artifacts else "X_test"

    if x_train_key == "X_train_selected":
        console.print("[dim]Using feature-selected splits.[/dim]\n")
    else:
        console.print("[dim]Feature selection not run — using all prepared features.[/dim]\n")

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
# Model selection
# ---------------------------------------------------------------------------

def _select_model(available: list[dict], state: dict) -> dict | None:
    existing_name = state.get("modeling", {}).get("model_name")
    choices = [m["name"] for m in available]
    default = existing_name if existing_name in choices else choices[0]

    name = questionary.select(
        "Select a model:",
        choices=choices,
        default=default,
    ).ask()
    if name is None:
        return None

    return next(m for m in available if m["name"] == name)


# ---------------------------------------------------------------------------
# Parameter configuration
# ---------------------------------------------------------------------------

def _configure_params(model_cfg: dict) -> dict | None:
    defaults = dict(model_cfg["default_params"])

    if not defaults:
        console.print("[dim]No parameters to configure for this model.[/dim]\n")
        return defaults

    console.print("\n[bold]Default Parameters[/bold]\n")
    table = Table(box=box.SIMPLE, header_style="bold blue")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")
    for k, v in defaults.items():
        table.add_row(k, str(v))
    console.print(table)

    accept = questionary.confirm("Accept default parameters?", default=True).ask()
    if accept is None:
        return None
    if accept:
        return defaults

    params = dict(defaults)
    for key, current_val in defaults.items():
        new_val_str = questionary.text(
            f"  {key} [{current_val}]:",
            default=str(current_val) if current_val is not None else "None",
        ).ask()
        if new_val_str is None:
            return None
        params[key] = _parse_param(new_val_str.strip())

    return params


def _parse_param(value: str):
    """Parse a string param value into the appropriate Python type."""
    if value.lower() == "none":
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def _train(model_cfg: dict, params: dict, X_train: pd.DataFrame, y_train: pd.Series):
    module_path, class_name = model_cfg["sklearn_class"].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
        model = ModelClass(**params)
    except Exception as e:
        console.print(f"[red]Failed to instantiate model: {e}[/red]")
        return None

    console.print(f"[blue]Training {model_cfg['name']}...[/blue]")
    try:
        model.fit(X_train, y_train)
        console.print("[green]Training complete.[/green]\n")
        return model
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        return None


# ---------------------------------------------------------------------------
# Quick eval
# ---------------------------------------------------------------------------

def _quick_eval(model, X_train, X_test, y_train, y_test, problem_type: str) -> dict:
    metrics: dict = {}

    try:
        if problem_type == "classification":
            from sklearn.metrics import accuracy_score
            train_acc = float(accuracy_score(y_train, model.predict(X_train)))
            test_acc = float(accuracy_score(y_test, model.predict(X_test)))
            metrics = {"train_accuracy": round(train_acc, 4), "test_accuracy": round(test_acc, 4)}

            console.print("[bold]Quick Evaluation[/bold]")
            console.print(f"  Train accuracy: [cyan]{train_acc:.4f}[/cyan]")
            console.print(f"  Test accuracy:  [cyan]{test_acc:.4f}[/cyan]")

            gap = train_acc - test_acc
            if gap > 0.1:
                console.print(f"  [yellow]Large train/test gap ({gap:.3f}) — possible overfitting.[/yellow]")
        else:
            from sklearn.metrics import mean_absolute_error, r2_score
            train_r2 = float(r2_score(y_train, model.predict(X_train)))
            test_r2 = float(r2_score(y_test, model.predict(X_test)))
            test_mae = float(mean_absolute_error(y_test, model.predict(X_test)))
            metrics = {
                "train_r2": round(train_r2, 4),
                "test_r2": round(test_r2, 4),
                "test_mae": round(test_mae, 4),
            }

            console.print("[bold]Quick Evaluation[/bold]")
            console.print(f"  Train R²:  [cyan]{train_r2:.4f}[/cyan]")
            console.print(f"  Test R²:   [cyan]{test_r2:.4f}[/cyan]")
            console.print(f"  Test MAE:  [cyan]{test_mae:.4f}[/cyan]")

            gap = train_r2 - test_r2
            if gap > 0.1:
                console.print(f"  [yellow]Large train/test gap ({gap:.3f}) — possible overfitting.[/yellow]")

        console.print()
    except Exception as e:
        console.print(f"[yellow]Could not compute metrics: {e}[/yellow]\n")

    return metrics


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save(model, model_cfg: dict, params: dict, metrics: dict, state: dict) -> dict:
    project_dir = Path(state["project"]["project_dir"])
    model_path = project_dir / "artifacts" / "model.joblib"

    joblib.dump(model, model_path)
    console.print("[green]Model saved to artifacts/model.joblib[/green]")

    summary = {
        "completed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "model_name": model_cfg["name"],
        "model_class": model_cfg["sklearn_class"],
        "params": params,
        "quick_metrics": metrics,
        "artifacts": {
            "model": "artifacts/model.joblib",
        },
    }

    if "artifacts" not in state:
        state["artifacts"] = {}
    state["artifacts"]["model"] = "artifacts/model.joblib"

    state = update_state(state, modeling=summary)
    state["stages"]["modeling"] = "completed"
    save_state(state)

    console.print("[green]Baseline model training complete.[/green]\n")
    return state
