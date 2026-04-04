import json
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd
import questionary
import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ml_orchestrator.deps.checker import check_and_install
from ml_orchestrator.state.manager import save_state, update_state
from ml_orchestrator.ui import charts

console = Console()

REQUIRED_PACKAGES: list[str] = []
OPTIONAL_PACKAGES: list[dict] = [
    {
        "package": "charset-normalizer",
        "reason": "Auto-detect file encoding for non-UTF-8 files",
    },
    {
        "package": "ydata-profiling",
        "reason": "Generate full HTML profiling report (requires pandas < 3.0 — may not install)",
    },
]

LEAKAGE_PATTERNS = {"target", "label", "outcome", "result", "response", "dependent"}


def run(state: dict) -> dict:
    console.print("\n[bold blue]── Profile Dataset ──[/bold blue]\n")

    if state.get("stages", {}).get("ingestion") != "completed":
        console.print("[yellow]No dataset loaded. Please run 'Read Dataset' first (option 3).[/yellow]")
        return state

    can_proceed, state = check_and_install(REQUIRED_PACKAGES, OPTIONAL_PACKAGES, state)
    if not can_proceed:
        return state

    df = _load_dataframe(state)
    if df is None:
        return state

    # Apply any renames from a prior data dictionary run
    df = _apply_renames(df, state)

    state = _data_dictionary(df, state)
    # Re-apply renames after data dictionary in case user added new ones
    df = _apply_renames(df, state)

    warnings = _profile(df, state)
    _show_null_chart(df)
    state = _offer_html_report(df, state)
    state = _save(df, warnings, state)

    return state


# ---------------------------------------------------------------------------
# Data Dictionary
# ---------------------------------------------------------------------------

def _data_dictionary(df: pd.DataFrame, state: dict) -> dict:
    console.print(Panel(
        "The data dictionary lets you describe this dataset and each column.\n"
        "Press [bold]Enter[/bold] to skip any field.",
        title="Data Dictionary",
        border_style="cyan",
    ))

    existing = state.get("data_dictionary", {})

    dataset_desc = questionary.text(
        "Dataset description:",
        default=existing.get("dataset_description", ""),
    ).ask()
    if dataset_desc is None:
        return state

    while True:
        columns_meta = dict(existing.get("columns", {}))

        for col in df.columns:
            col_meta = columns_meta.get(str(col), {})
            sample = _sample_value(df[col])

            console.print(
                f"\n  [cyan]{col}[/cyan]  "
                f"[dim]{df[col].dtype}[/dim]  "
                f"sample: [italic]{sample}[/italic]"
            )

            rename = questionary.text(
                "  Rename to (Enter to keep):",
                default=col_meta.get("renamed_to") or "",
            ).ask()
            if rename is None:
                return state

            description = questionary.text(
                "  Description:",
                default=col_meta.get("description") or "",
            ).ask()
            if description is None:
                return state

            columns_meta[str(col)] = {
                "renamed_to": rename.strip() or None,
                "description": description.strip() or None,
            }

        # Show summary and confirm
        console.print("\n[bold]Data Dictionary Summary[/bold]\n")
        summary_table = Table(box=box.SIMPLE, header_style="bold blue")
        summary_table.add_column("Column", style="cyan")
        summary_table.add_column("Renamed To")
        summary_table.add_column("Description")
        for col, meta in columns_meta.items():
            summary_table.add_row(
                col,
                meta.get("renamed_to") or "[dim]—[/dim]",
                meta.get("description") or "[dim]—[/dim]",
            )
        console.print(summary_table)

        confirm = questionary.confirm(
            "Apply these changes?",
            default=True,
        ).ask()
        if confirm is None:
            return state
        if confirm:
            break
        console.print("\n[yellow]Starting over — re-enter column details.[/yellow]\n")

    data_dictionary = {
        "dataset_description": dataset_desc.strip() or None,
        "columns": columns_meta,
    }

    # Export to artifacts
    project_dir = Path(state["project"]["project_dir"])
    dd_path = project_dir / "artifacts" / "data_dictionary.yaml"
    with open(dd_path, "w", encoding="utf-8") as f:
        yaml.dump(data_dictionary, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    console.print(f"\n[green]Data dictionary saved to {dd_path.relative_to(project_dir)}[/green]\n")

    state = update_state(state, data_dictionary=data_dictionary)
    if "artifacts" not in state:
        state["artifacts"] = {}
    state["artifacts"]["data_dictionary"] = "artifacts/data_dictionary.yaml"

    return state


def _apply_renames(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Rename columns in the dataframe based on data_dictionary state."""
    dd = state.get("data_dictionary", {})
    rename_map = {}
    for col, meta in dd.get("columns", {}).items():
        if meta and meta.get("renamed_to"):
            rename_map[col] = meta["renamed_to"]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ---------------------------------------------------------------------------
# Power BI-style profiling
# ---------------------------------------------------------------------------

def _profile(df: pd.DataFrame, state: dict) -> list[dict]:
    n_rows = len(df)
    n_dupes = int(df.duplicated().sum())

    console.print("[bold]Dataset Overview[/bold]")
    console.print(f"  Rows: [cyan]{n_rows:,}[/cyan]  |  "
                  f"Columns: [cyan]{len(df.columns)}[/cyan]  |  "
                  f"Duplicate rows: [cyan]{n_dupes:,}[/cyan]\n")

    settings = state.get("settings", {})
    high_card_threshold = settings.get("high_cardinality_threshold", 50)
    imbalance_threshold = settings.get("class_imbalance_threshold", 0.75)
    target_col = state.get("feature_preparation", {}) and state.get("feature_preparation", {}).get("target_column")

    warnings: list[dict] = []

    table = Table(
        title="Column Profiles",
        box=box.SIMPLE,
        header_style="bold blue",
        show_lines=True,
    )
    table.add_column("Column", style="cyan", no_wrap=True)
    table.add_column("Type", justify="center")
    table.add_column("Non-Null", justify="right")
    table.add_column("Null %", justify="right")
    table.add_column("Distinct", justify="right")
    table.add_column("Distinct %", justify="right")
    table.add_column("Stats / Top Values", style="dim")

    for col in df.columns:
        series = df[col]
        n_null = int(series.isnull().sum())
        null_pct = n_null / n_rows * 100 if n_rows > 0 else 0
        n_distinct = int(series.nunique(dropna=True))
        distinct_pct = n_distinct / n_rows * 100 if n_rows > 0 else 0
        non_null = n_rows - n_null

        # Stats column
        if pd.api.types.is_numeric_dtype(series):
            stats = (
                f"min={series.min():.3g}  max={series.max():.3g}  "
                f"mean={series.mean():.3g}  std={series.std():.3g}"
            )
        else:
            top = series.value_counts().head(3)
            parts = [f"{v}({c})" for v, c in top.items()]
            stats = "  ".join(parts)

        # Null % colour
        null_str = f"{null_pct:.1f}%"
        if null_pct > 50:
            null_str = f"[red]{null_str}[/red]"
        elif null_pct > 10:
            null_str = f"[yellow]{null_str}[/yellow]"

        table.add_row(
            str(col),
            str(series.dtype),
            f"{non_null:,}",
            null_str,
            f"{n_distinct:,}",
            f"{distinct_pct:.1f}%",
            stats,
        )

        # --- Collect warnings ---
        if n_distinct == 1:
            warnings.append({"type": "constant_column", "column": str(col),
                             "detail": "Only one unique value — not useful for modeling"})
        elif n_distinct == n_rows and not pd.api.types.is_numeric_dtype(series):
            warnings.append({"type": "id_like_column", "column": str(col),
                             "detail": "Unique count equals row count — likely an ID column"})

        if not pd.api.types.is_numeric_dtype(series) and n_distinct > high_card_threshold:
            warnings.append({"type": "high_cardinality", "column": str(col),
                             "detail": f"{n_distinct} unique values exceeds threshold ({high_card_threshold})"})

        if null_pct > 50:
            warnings.append({"type": "high_null_rate", "column": str(col),
                             "detail": f"{null_pct:.1f}% null values"})

        col_lower = str(col).lower()
        if any(p in col_lower for p in LEAKAGE_PATTERNS) and str(col) != target_col:
            warnings.append({"type": "possible_leakage", "column": str(col),
                             "detail": f"Column name matches leakage pattern: '{col_lower}'"})

        if str(col) == target_col and not pd.api.types.is_numeric_dtype(series):
            top_freq = series.value_counts(normalize=True).iloc[0] if len(series) > 0 else 0
            if top_freq > imbalance_threshold:
                warnings.append({"type": "class_imbalance", "column": str(col),
                                 "detail": f"Majority class is {top_freq:.1%} of target"})

    console.print(table)

    # Show warnings
    if warnings:
        _show_warnings(warnings)
    else:
        console.print("[green]No data quality warnings detected.[/green]\n")

    # Optional: drill into a column
    _offer_column_drill(df)

    return warnings


def _show_warnings(warnings: list[dict]) -> None:
    console.print("\n[bold]Data Quality Warnings[/bold]\n")
    for w in warnings:
        icon = "[red]●[/red]" if w["type"] in ("constant_column", "id_like_column") else "[yellow]●[/yellow]"
        console.print(f"  {icon}  [bold]{w['column']}[/bold] — {w['detail']}")
    console.print()


def _offer_column_drill(df: pd.DataFrame) -> None:
    while True:
        drill = questionary.confirm(
            "Drill into a specific column for distribution chart?",
            default=False,
        ).ask()
        if not drill:
            return

        col_name = questionary.select(
            "Select column:",
            choices=[str(c) for c in df.columns],
        ).ask()
        if col_name is None:
            return

        series = df[col_name]
        if pd.api.types.is_numeric_dtype(series):
            counts, edges = pd.cut(series.dropna(), bins=10, retbins=True)
            labels = [f"{e:.2g}" for e in edges[:-1]]
            values = [float(v) for v in counts.value_counts().sort_index().values]
            charts.bar(labels, values, title=f"Distribution: {col_name}", unit="count")
        else:
            top = series.value_counts().head(15)
            charts.horizontal_bar(
                labels=[str(v) for v in top.index],
                values=[float(v) for v in top.values],
                title=f"Top values: {col_name}",
                unit="count",
            )


# ---------------------------------------------------------------------------
# Null chart
# ---------------------------------------------------------------------------

def _show_null_chart(df: pd.DataFrame) -> None:
    null_pcts = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    null_pcts = null_pcts[null_pcts > 0].head(15)

    if null_pcts.empty:
        console.print("[green]No missing values found in this dataset.[/green]\n")
        return

    console.print("\n[bold]Missing Values Chart[/bold]")
    charts.horizontal_bar(
        labels=[str(c) for c in null_pcts.index],
        values=[round(float(v), 2) for v in null_pcts.values],
        title="Null % by Column (top 15)",
        unit="% null",
    )


# ---------------------------------------------------------------------------
# Optional HTML report
# ---------------------------------------------------------------------------

def _offer_html_report(df: pd.DataFrame, state: dict) -> dict:
    try:
        from ydata_profiling import ProfileReport
        has_ydata = True
    except ImportError:
        has_ydata = False

    if not has_ydata:
        return state

    generate = questionary.confirm(
        "Generate full HTML profiling report and open in browser?",
        default=False,
    ).ask()
    if not generate:
        return state

    project_dir = Path(state["project"]["project_dir"])
    report_path = project_dir / "artifacts" / "profiling_report.html"

    console.print("[blue]Generating HTML report (this may take a moment)...[/blue]")
    profile = ProfileReport(df, title="Dataset Profile", minimal=True)
    profile.to_file(report_path)

    webbrowser.open(report_path.as_uri())
    console.print(f"[green]Report saved to {report_path.relative_to(project_dir)}[/green]")

    state["artifacts"]["profiling_report"] = "artifacts/profiling_report.html"
    return state


# ---------------------------------------------------------------------------
# Save to state
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, warnings: list[dict], state: dict) -> dict:
    project_dir = Path(state["project"]["project_dir"])
    n_rows = len(df)

    null_counts = {
        col: int(df[col].isnull().sum())
        for col in df.columns
        if df[col].isnull().sum() > 0
    }

    summary = {
        "completed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "row_count": n_rows,
        "duplicate_rows": int(df.duplicated().sum()),
        "null_counts": null_counts,
        "warnings": warnings,
        "artifacts": {
            "json_summary": "artifacts/profiling_summary.json",
        },
    }

    # Export JSON summary
    summary_path = project_dir / "artifacts" / "profiling_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if "artifacts" not in state:
        state["artifacts"] = {}
    state["artifacts"]["profiling_summary"] = "artifacts/profiling_summary.json"

    state = update_state(state, profiling=summary)
    state["stages"]["profiling"] = "completed"
    save_state(state)

    console.print("[green]Profiling complete. Results saved to project state.[/green]\n")
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataframe(state: dict) -> pd.DataFrame | None:
    source_path = state.get("ingestion", {}).get("source_path")
    if not source_path:
        console.print("[red]No dataset path found in state. Please run ingestion first.[/red]")
        return None

    path = Path(source_path)
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        else:
            encoding = state["ingestion"].get("encoding", "utf-8")
            delimiter = state["ingestion"].get("delimiter", ",")
            return pd.read_csv(path, encoding=encoding, sep=delimiter)
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        return None


def _sample_value(series: pd.Series) -> str:
    sample = series.dropna()
    if sample.empty:
        return "—"
    val = str(sample.iloc[0])
    return val[:40] + "…" if len(val) > 40 else val
