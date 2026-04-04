import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import questionary
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ml_orchestrator.deps.checker import check_and_install
from ml_orchestrator.state.manager import save_state, update_state
from ml_orchestrator.ui import charts

console = Console()

REQUIRED_PACKAGES: list[str] = ["scikit-learn"]
OPTIONAL_PACKAGES: list[dict] = []


def run(state: dict) -> dict:
    console.print("\n[bold blue]── Evaluate Model ──[/bold blue]\n")

    if state.get("stages", {}).get("modeling") != "completed":
        console.print("[yellow]No model found. Please run 'Train Baseline Model' first (option 8).[/yellow]")
        return state

    can_proceed, state = check_and_install(REQUIRED_PACKAGES, OPTIONAL_PACKAGES, state)
    if not can_proceed:
        return state

    model, model_key = _load_model(state)
    if model is None:
        return state

    X_test, y_test = _load_test_split(state)
    if X_test is None:
        return state

    problem_type = state.get("feature_preparation", {}).get("problem_type", "classification")

    if problem_type == "classification":
        metrics, report = _evaluate_classification(model, X_test, y_test)
    else:
        metrics, report = _evaluate_regression(model, X_test, y_test)

    if metrics is None:
        return state

    state = _save(metrics, report, model_key, problem_type, state)
    return state


# ---------------------------------------------------------------------------
# Load model and data
# ---------------------------------------------------------------------------

def _load_model(state: dict) -> tuple:
    project_dir = Path(state["project"]["project_dir"])
    artifacts = state.get("artifacts", {})

    if "model_tuned" in artifacts:
        key = "model_tuned"
        console.print("[dim]Using tuned model (model_tuned.joblib).[/dim]\n")
    else:
        key = "model"
        console.print("[dim]Using baseline model (model.joblib).[/dim]\n")

    try:
        model = joblib.load(project_dir / artifacts[key])
        return model, key
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        return None, None


def _load_test_split(state: dict) -> tuple:
    project_dir = Path(state["project"]["project_dir"])
    artifacts = state.get("artifacts", {})

    x_test_key = "X_test_selected" if "X_test_selected" in artifacts else "X_test"

    try:
        X_test = pd.read_csv(project_dir / artifacts[x_test_key])
        y_test = pd.read_csv(project_dir / artifacts["y_test"]).squeeze()
        return X_test, y_test
    except Exception as e:
        console.print(f"[red]Failed to load test split: {e}[/red]")
        return None, None


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def _evaluate_classification(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        console.print(f"[red]Prediction failed: {e}[/red]")
        return None, None

    classes = sorted(y_test.unique())
    is_binary = len(classes) == 2

    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    metrics: dict = {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
    }

    # ROC-AUC for binary
    roc_auc = None
    if is_binary and hasattr(model, "predict_proba"):
        from sklearn.metrics import roc_auc_score
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = round(float(roc_auc_score(y_test, y_prob)), 4)
            metrics["roc_auc"] = roc_auc
        except Exception:
            pass

    # Per-class report
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # --- Display ---
    console.print(Panel("[bold]Classification Results[/bold]", border_style="blue"))

    # Headline metrics
    headline = Table(box=box.SIMPLE, show_header=False)
    headline.add_column(style="dim")
    headline.add_column(style="cyan bold")
    headline.add_row("Accuracy", f"{accuracy:.4f}")
    headline.add_row("Precision (macro)", f"{precision_macro:.4f}")
    headline.add_row("Recall (macro)", f"{recall_macro:.4f}")
    headline.add_row("F1 (macro)", f"{f1_macro:.4f}")
    headline.add_row("F1 (weighted)", f"{f1_weighted:.4f}")
    if roc_auc is not None:
        headline.add_row("ROC-AUC", f"{roc_auc:.4f}")
    console.print(headline)

    # Per-class table
    console.print("\n[bold]Per-Class Report[/bold]\n")
    cls_table = Table(box=box.SIMPLE, header_style="bold blue")
    cls_table.add_column("Class", style="cyan")
    cls_table.add_column("Precision", justify="right")
    cls_table.add_column("Recall", justify="right")
    cls_table.add_column("F1", justify="right")
    cls_table.add_column("Support", justify="right")
    for cls in classes:
        row = report_dict.get(str(cls), {})
        cls_table.add_row(
            str(cls),
            f"{row.get('precision', 0):.3f}",
            f"{row.get('recall', 0):.3f}",
            f"{row.get('f1-score', 0):.3f}",
            str(int(row.get("support", 0))),
        )
    console.print(cls_table)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    _show_confusion_matrix(cm, classes)

    # F1 bar chart per class
    f1_scores = [report_dict.get(str(c), {}).get("f1-score", 0) for c in classes]
    if len(classes) > 1:
        charts.horizontal_bar(
            [str(c) for c in classes],
            [round(s, 4) for s in f1_scores],
            title="F1 Score per Class",
            unit="f1",
        )

    return metrics, report_dict


def _show_confusion_matrix(cm, classes) -> None:
    console.print("\n[bold]Confusion Matrix[/bold]  [dim](rows = actual, cols = predicted)[/dim]\n")
    table = Table(box=box.SIMPLE, header_style="bold blue")
    table.add_column("Actual \\ Pred", style="dim")
    for cls in classes:
        table.add_column(str(cls), justify="right")

    for i, cls in enumerate(classes):
        row_vals = []
        for j in range(len(classes)):
            val = int(cm[i][j])
            if i == j:
                row_vals.append(f"[green]{val}[/green]")
            else:
                row_vals.append(f"[red]{val}[/red]" if val > 0 else str(val))
        table.add_row(str(cls), *row_vals)

    console.print(table)


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------

def _evaluate_regression(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        console.print(f"[red]Prediction failed: {e}[/red]")
        return None, None

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = math.sqrt(mse)

    # MAPE — guard against zero actuals
    y_arr = np.array(y_test, dtype=float)
    nonzero = y_arr != 0
    mape = float(np.mean(np.abs((y_arr[nonzero] - y_pred[nonzero]) / y_arr[nonzero]))) * 100 if nonzero.any() else None

    metrics: dict = {
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
    }
    if mape is not None:
        metrics["mape_pct"] = round(mape, 2)

    # --- Display ---
    console.print(Panel("[bold]Regression Results[/bold]", border_style="blue"))

    headline = Table(box=box.SIMPLE, show_header=False)
    headline.add_column(style="dim")
    headline.add_column(style="cyan bold")
    headline.add_row("R²", f"{r2:.4f}")
    headline.add_row("MAE", f"{mae:.4f}")
    headline.add_row("RMSE", f"{rmse:.4f}")
    headline.add_row("MSE", f"{mse:.4f}")
    if mape is not None:
        headline.add_row("MAPE", f"{mape:.2f}%")
    console.print(headline)

    # Residuals distribution (binned into 10 buckets)
    residuals = np.array(y_test, dtype=float) - y_pred
    _show_residuals_chart(residuals)

    return metrics, {}


def _show_residuals_chart(residuals: np.ndarray) -> None:
    n_bins = 10
    counts, bin_edges = np.histogram(residuals, bins=n_bins)
    labels = [f"{bin_edges[i]:.1f}" for i in range(n_bins)]
    charts.bar(labels, list(counts.astype(float)), title="Residuals Distribution", unit="count")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save(
    metrics: dict,
    report: dict,
    model_key: str,
    problem_type: str,
    state: dict,
) -> dict:
    project_dir = Path(state["project"]["project_dir"])

    save_report = questionary.confirm("Save evaluation report to artifacts?", default=True).ask()

    report_path_rel = None
    if save_report:
        full_report = {
            "model_used": model_key,
            "problem_type": problem_type,
            "metrics": metrics,
            "classification_report": report if problem_type == "classification" else {},
            "evaluated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        report_path = project_dir / "artifacts" / "evaluation_report.json"
        report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
        report_path_rel = "artifacts/evaluation_report.json"
        console.print("[green]Evaluation report saved to artifacts/evaluation_report.json[/green]")

        if "artifacts" not in state:
            state["artifacts"] = {}
        state["artifacts"]["evaluation_report"] = report_path_rel

    summary = {
        "completed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "model_used": model_key,
        "problem_type": problem_type,
        "metrics": metrics,
        "artifacts": {"evaluation_report": report_path_rel} if report_path_rel else {},
    }

    state = update_state(state, evaluation=summary)
    state["stages"]["evaluation"] = "completed"
    save_state(state)

    console.print("[green]Evaluation complete.[/green]\n")
    return state
