from datetime import datetime
from pathlib import Path

import pandas as pd
import questionary
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ml_orchestrator.state.manager import save_state, update_state

console = Console()


def run(state: dict) -> dict:
    console.print("\n[bold blue]── Clean Dataset ──[/bold blue]\n")

    if state.get("stages", {}).get("ingestion") != "completed":
        console.print("[yellow]No dataset loaded. Please run 'Read Dataset' first (option 3).[/yellow]")
        return state

    df = _load_dataframe(state)
    if df is None:
        return state

    df = _apply_renames(df, state)

    actions: list[dict] = []
    excluded_columns: list[str] = []

    df, actions = _handle_duplicates(df, actions)
    df, actions, excluded_columns = _handle_columns(df, state, actions, excluded_columns)
    state = _review_and_save(df, actions, excluded_columns, state)

    return state


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def _handle_duplicates(df: pd.DataFrame, actions: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    n_dupes = int(df.duplicated().sum())

    if n_dupes == 0:
        console.print("[green]No duplicate rows found.[/green]\n")
        actions.append({"type": "remove_duplicates", "rows_removed": 0})
        return df, actions

    console.print(f"[yellow]Found {n_dupes:,} duplicate rows.[/yellow]")
    remove = questionary.confirm(f"Remove {n_dupes:,} duplicate rows?", default=True).ask()
    if remove is None:
        return df, actions

    if remove:
        df = df.drop_duplicates().reset_index(drop=True)
        actions.append({"type": "remove_duplicates", "rows_removed": n_dupes})
        console.print(f"[green]Removed {n_dupes:,} duplicate rows.[/green]\n")
    else:
        actions.append({"type": "remove_duplicates", "rows_removed": 0})

    return df, actions


# ---------------------------------------------------------------------------
# Per-column attention
# ---------------------------------------------------------------------------

def _handle_columns(
    df: pd.DataFrame,
    state: dict,
    actions: list[dict],
    excluded_columns: list[str],
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    profiling_warnings = state.get("profiling", {}).get("warnings", [])

    # Map column -> list of warnings (only for columns still in df)
    warn_map: dict[str, list[dict]] = {}
    for w in profiling_warnings:
        col = w.get("column")
        if col and col in df.columns:
            warn_map.setdefault(col, []).append(w)

    null_cols = [col for col in df.columns if df[col].isnull().any()]
    warn_only_cols = [col for col in warn_map if col not in null_cols]
    attention_cols = null_cols + warn_only_cols

    if not attention_cols:
        console.print("[green]No columns require attention.[/green]\n")
        return df, actions, excluded_columns

    console.print(Panel(
        f"[bold]{len(attention_cols)}[/bold] column(s) need attention — reviewing each in turn.",
        border_style="yellow",
    ))

    for col in attention_cols:
        if col not in df.columns:
            continue

        col_warnings = warn_map.get(col, [])
        warn_types = {w["type"] for w in col_warnings}
        n_null = int(df[col].isnull().sum())
        n_rows = len(df)
        null_pct = n_null / n_rows * 100 if n_rows > 0 else 0
        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        # Column header
        console.print(f"\n[bold cyan]{col}[/bold cyan]  [dim]{df[col].dtype}[/dim]")
        if n_null > 0:
            color = "red" if null_pct > 50 else "yellow"
            console.print(f"  [{color}]{n_null:,} nulls ({null_pct:.1f}%)[/{color}]")
        for w in col_warnings:
            icon = "[red]●[/red]" if w["type"] in ("constant_column", "id_like_column") else "[yellow]●[/yellow]"
            console.print(f"  {icon} {w['detail']}")

        # Route to appropriate prompt
        if "constant_column" in warn_types or "id_like_column" in warn_types:
            action = _prompt_exclude_or_keep(col, col_warnings, default_exclude=True)
        elif "high_null_rate" in warn_types or null_pct > 50:
            action = _prompt_high_null(col, is_numeric, df)
        elif n_null > 0:
            action = _prompt_nulls(col, is_numeric, df)
        else:
            # Warning only (high cardinality, possible leakage) — no imputation needed
            action = _prompt_exclude_or_keep(col, col_warnings, default_exclude=False)

        if action is None:
            continue

        if action["type"] == "exclude_column":
            excluded_columns.append(col)
            df = df.drop(columns=[col])
            actions.append(action)
        elif action["type"] == "impute":
            df = _apply_impute(df, col, action)
            actions.append(action)
        elif action["type"] == "drop_rows_with_nulls":
            before = len(df)
            df = df.dropna(subset=[col])
            action["rows_removed"] = before - len(df)
            actions.append(action)
        # "keep" → no action recorded

    return df, actions, excluded_columns


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _prompt_exclude_or_keep(
    col: str,
    warnings: list[dict],
    default_exclude: bool,
) -> dict | None:
    choices = ["Exclude this column", "Keep as-is"]
    default = choices[0] if default_exclude else choices[1]
    choice = questionary.select(
        f"  Action for '{col}':",
        choices=choices,
        default=default,
    ).ask()
    if choice is None:
        return None
    if choice.startswith("Exclude"):
        reason = warnings[0]["type"] if warnings else "user decision"
        return {"type": "exclude_column", "column": col, "reason": reason}
    return None  # keep → no action


def _prompt_high_null(col: str, is_numeric: bool, df: pd.DataFrame) -> dict | None:
    choices = ["Exclude this column (recommended)", "Drop rows with nulls"]
    if is_numeric:
        choices += ["Impute with mean", "Impute with median", "Impute with constant"]
    else:
        choices += ["Impute with mode", "Impute with constant"]
    choices.append("Keep as-is")
    choice = questionary.select(f"  Action for '{col}':", choices=choices, default=choices[0]).ask()
    return _parse_impute_choice(choice, col, df)


def _prompt_nulls(col: str, is_numeric: bool, df: pd.DataFrame) -> dict | None:
    if is_numeric:
        choices = [
            "Impute with median (recommended)",
            "Impute with mean",
            "Impute with constant",
            "Drop rows with nulls",
            "Exclude this column",
            "Keep as-is",
        ]
    else:
        choices = [
            "Impute with mode (recommended)",
            "Impute with constant",
            "Drop rows with nulls",
            "Exclude this column",
            "Keep as-is",
        ]
    choice = questionary.select(f"  Action for '{col}':", choices=choices, default=choices[0]).ask()
    return _parse_impute_choice(choice, col, df)


def _parse_impute_choice(choice: str | None, col: str, df: pd.DataFrame) -> dict | None:
    if choice is None:
        return None
    if "Exclude" in choice:
        return {"type": "exclude_column", "column": col, "reason": "user decision"}
    if "Drop rows" in choice:
        return {"type": "drop_rows_with_nulls", "column": col}
    if "mean" in choice.lower():
        return {"type": "impute", "column": col, "strategy": "mean", "value": float(df[col].mean())}
    if "median" in choice.lower():
        return {"type": "impute", "column": col, "strategy": "median", "value": float(df[col].median())}
    if "mode" in choice.lower():
        mode_vals = df[col].mode()
        val = str(mode_vals.iloc[0]) if not mode_vals.empty else ""
        return {"type": "impute", "column": col, "strategy": "mode", "value": val}
    if "constant" in choice.lower():
        constant = questionary.text(f"  Fill value for '{col}':").ask()
        if constant is None:
            return None
        return {"type": "impute", "column": col, "strategy": "constant", "value": constant.strip()}
    return None  # keep as-is


# ---------------------------------------------------------------------------
# Apply imputation
# ---------------------------------------------------------------------------

def _apply_impute(df: pd.DataFrame, col: str, action: dict) -> pd.DataFrame:
    strategy = action["strategy"]
    value = action["value"]
    if strategy in ("mean", "median"):
        df[col] = df[col].fillna(float(value))
    elif strategy == "mode":
        df[col] = df[col].fillna(value)
    elif strategy == "constant":
        try:
            df[col] = df[col].fillna(float(value))
        except (ValueError, TypeError):
            df[col] = df[col].fillna(str(value))
    return df


# ---------------------------------------------------------------------------
# Review and save
# ---------------------------------------------------------------------------

def _review_and_save(
    df: pd.DataFrame,
    actions: list[dict],
    excluded_columns: list[str],
    state: dict,
) -> dict:
    meaningful_actions = [a for a in actions if not (a["type"] == "remove_duplicates" and a["rows_removed"] == 0)]

    console.print("\n[bold]Cleaning Summary[/bold]\n")
    if not meaningful_actions:
        console.print("[green]No changes to apply.[/green]\n")
    else:
        summary_table = Table(box=box.SIMPLE, header_style="bold blue")
        summary_table.add_column("Action")
        summary_table.add_column("Column")
        summary_table.add_column("Detail")
        for a in actions:
            if a["type"] == "remove_duplicates" and a["rows_removed"] == 0:
                continue
            detail = ""
            col = a.get("column", "—")
            if a["type"] == "remove_duplicates":
                col = "—"
                detail = f"{a['rows_removed']:,} rows removed"
            elif a["type"] == "exclude_column":
                detail = a.get("reason", "")
            elif a["type"] == "impute":
                detail = f"strategy={a['strategy']}, fill={a.get('value', '')}"
            elif a["type"] == "drop_rows_with_nulls":
                detail = f"{a.get('rows_removed', 0):,} rows removed"
            summary_table.add_row(a["type"], col, str(detail))
        console.print(summary_table)

    console.print(f"Rows remaining: [cyan]{len(df):,}[/cyan]  |  Columns remaining: [cyan]{len(df.columns)}[/cyan]\n")

    confirm = questionary.confirm("Save cleaned dataset?", default=True).ask()
    if not confirm:
        console.print("[yellow]Cleaning cancelled — no changes saved.[/yellow]\n")
        return state

    project_dir = Path(state["project"]["project_dir"])
    out_path = project_dir / "artifacts" / "cleaned_dataset.csv"
    df.to_csv(out_path, index=False)
    console.print(f"[green]Cleaned dataset saved to {out_path.relative_to(project_dir)}[/green]")

    cleaning = {
        "completed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "actions": [a for a in actions if not (a["type"] == "remove_duplicates" and a["rows_removed"] == 0)],
        "excluded_columns": excluded_columns,
        "row_count": len(df),
        "column_count": len(df.columns),
        "artifacts": {
            "cleaned_dataset": "artifacts/cleaned_dataset.csv",
        },
    }

    if "artifacts" not in state:
        state["artifacts"] = {}
    state["artifacts"]["cleaned_dataset"] = "artifacts/cleaned_dataset.csv"

    state = update_state(state, cleaning=cleaning)
    state["stages"]["cleaning"] = "completed"
    save_state(state)

    console.print("[green]Cleaning complete. Results saved to project state.[/green]\n")
    return state


# ---------------------------------------------------------------------------
# Helpers (shared pattern with profiling)
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


def _apply_renames(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    dd = state.get("data_dictionary", {})
    rename_map = {}
    for col, meta in dd.get("columns", {}).items():
        if meta and meta.get("renamed_to"):
            rename_map[col] = meta["renamed_to"]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df
