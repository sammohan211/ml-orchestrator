import csv
from pathlib import Path

import pandas as pd
import questionary
from rich.console import Console
from rich.table import Table
from rich import box
from rich.status import Status

from ml_orchestrator.deps.checker import check_and_install
from ml_orchestrator.formats.registry import get_format, supported_extensions
from ml_orchestrator.state.manager import save_state, update_state

console = Console()

REQUIRED_PACKAGES: list[str] = []
OPTIONAL_PACKAGES: list[dict] = [
    {
        "package": "charset-normalizer",
        "reason": "Auto-detect file encoding for non-UTF-8 files",
    }
]


def run(state: dict) -> dict:
    """
    Ingestion stage entry point.
    Prompts user for a dataset file, auto-detects format settings,
    confirms with user, loads the file, displays schema and preview,
    saves to state.
    """
    console.print("\n[bold blue]── Read Dataset ──[/bold blue]\n")

    can_proceed, state = check_and_install(REQUIRED_PACKAGES, OPTIONAL_PACKAGES, state)
    if not can_proceed:
        return state

    # --- File path prompt ---
    exts = ", ".join(supported_extensions())
    path_str = questionary.text(
        f"Path to dataset file (supported: {exts}):",
        validate=lambda v: _validate_path(v),
    ).ask()

    if path_str is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return state

    path = Path(path_str.strip()).expanduser().resolve()
    fmt = get_format(path)

    if fmt is None:
        console.print(f"[red]Unsupported file format '{path.suffix}'. Supported: {exts}[/red]")
        return state

    # --- Format-specific loading ---
    if path.suffix.lower() == ".parquet":
        state = _load_parquet(path, fmt, state)
    else:
        state = _load_csv_interactive(path, fmt, state)

    return state


# ---------------------------------------------------------------------------
# CSV: auto-detect → show snippet → confirm → load
# ---------------------------------------------------------------------------

def _load_csv_interactive(path: Path, fmt, state: dict) -> dict:
    console.print(f"\n[dim]Inspecting file: {path}[/dim]")

    # Read raw bytes to detect encoding
    encoding = _detect_encoding(path)

    # Read first 20 raw lines for sniffing and display
    try:
        with open(path, encoding=encoding, errors="replace") as f:
            raw_lines = [f.readline() for _ in range(20)]
    except Exception as e:
        console.print(f"[red]Could not read file: {e}[/red]")
        return state

    sample_text = "".join(raw_lines)

    # Auto-detect delimiter and quote character
    delimiter, quotechar, has_header = _sniff_csv(sample_text)

    # Show raw snippet
    _show_raw_snippet(raw_lines[:6], encoding, delimiter)

    # Confirm or correct detected settings
    settings = _confirm_csv_settings(encoding, delimiter, has_header)
    if settings is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return state

    confirmed_encoding = settings["encoding"]
    confirmed_delimiter = settings["delimiter"]
    confirmed_header = settings["header"]

    # Load full file
    header_row = 0 if confirmed_header else None
    with Status("[blue]Loading dataset...[/blue]", console=console):
        try:
            df = pd.read_csv(
                path,
                encoding=confirmed_encoding,
                sep=confirmed_delimiter,
                header=header_row,
            )
        except Exception as e:
            console.print(f"[red]Failed to load file: {e}[/red]")
            return state

    return _finish(df, path, state, extra={"encoding": confirmed_encoding, "delimiter": confirmed_delimiter})


# ---------------------------------------------------------------------------
# Parquet: load directly
# ---------------------------------------------------------------------------

def _load_parquet(path: Path, fmt, state: dict) -> dict:
    with Status("[blue]Loading dataset...[/blue]", console=console):
        try:
            df = fmt.load(path)
        except Exception as e:
            console.print(f"[red]Failed to load file: {e}[/red]")
            return state

    return _finish(df, path, state)


# ---------------------------------------------------------------------------
# Post-load: schema display, preview, state update
# ---------------------------------------------------------------------------

def _finish(df: pd.DataFrame, path: Path, state: dict, extra: dict | None = None) -> dict:
    console.print(f"\n[green]Loaded:[/green] {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    _show_schema(df)
    _show_preview(df)

    # Build schema for state
    schema = {
        col: {
            "dtype": str(df[col].dtype),
            "nullable": bool(df[col].isnull().any()),
        }
        for col in df.columns
    }

    ingestion_data = {
        "ingestion": {
            "source_path": str(path),
            "format": path.suffix.lstrip(".").lower(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "schema": schema,
            **(extra or {}),
        }
    }

    state = update_state(state, **ingestion_data)
    state["stages"]["ingestion"] = "completed"
    save_state(state)

    console.print("\n[green]Dataset saved to project state.[/green]")
    return state


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_encoding(path: Path) -> str:
    """Detect file encoding using charset-normalizer if available, else default to utf-8."""
    try:
        from charset_normalizer import from_path
        result = from_path(path).best()
        if result:
            return result.encoding
    except ImportError:
        pass
    return "utf-8"


def _sniff_csv(sample: str) -> tuple[str, str, bool]:
    """Use csv.Sniffer to detect delimiter, quote char, and whether a header exists."""
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=",;\t|")
        has_header = sniffer.has_header(sample)
        return dialect.delimiter, dialect.quotechar, has_header
    except csv.Error:
        return ",", '"', True


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _show_raw_snippet(lines: list[str], encoding: str, delimiter: str) -> None:
    console.print(f"[bold]File preview[/bold] [dim](detected encoding: {encoding}, delimiter: {repr(delimiter)})[/dim]\n")
    for i, line in enumerate(lines, 1):
        console.print(f"  [dim]{i:>2}[/dim]  {line.rstrip()}")
    console.print()


def _confirm_csv_settings(encoding: str, delimiter: str, has_header: bool) -> dict | None:
    """Show detected CSV settings and let the user confirm or change each one."""
    console.print("[bold]Detected CSV settings[/bold] — press Enter to accept or type a new value.\n")

    confirmed_encoding = questionary.text(
        "Encoding:",
        default=encoding,
    ).ask()
    if confirmed_encoding is None:
        return None

    confirmed_delimiter = questionary.text(
        "Delimiter:",
        default=delimiter,
    ).ask()
    if confirmed_delimiter is None:
        return None

    confirmed_header = questionary.confirm(
        "Does the file have a header row?",
        default=has_header,
    ).ask()
    if confirmed_header is None:
        return None

    return {
        "encoding": confirmed_encoding.strip(),
        "delimiter": confirmed_delimiter,
        "header": confirmed_header,
    }


def _show_schema(df: pd.DataFrame) -> None:
    table = Table(title="Schema", box=box.SIMPLE, header_style="bold blue")
    table.add_column("Column", style="cyan")
    table.add_column("Type")
    table.add_column("Nullable", justify="center")
    table.add_column("Sample value", style="dim")

    for col in df.columns:
        nullable = "yes" if df[col].isnull().any() else "no"
        sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "—"
        sample = sample[:40] + "…" if len(sample) > 40 else sample
        table.add_row(str(col), str(df[col].dtype), nullable, sample)

    console.print(table)


def _show_preview(df: pd.DataFrame) -> None:
    table = Table(title="Data Preview (first 5 rows)", box=box.SIMPLE, header_style="bold blue")
    for col in df.columns:
        table.add_column(str(col), style="dim", no_wrap=True)

    for _, row in df.head(5).iterrows():
        values = [str(v)[:30] for v in row]
        table.add_row(*values)

    console.print(table)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_path(value: str) -> bool | str:
    p = Path(value.strip()).expanduser()
    if not p.exists():
        return "File not found."
    if not p.is_file():
        return "Path is a directory, not a file."
    if p.suffix.lower() not in supported_extensions():
        return f"Unsupported format '{p.suffix}'. Supported: {', '.join(supported_extensions())}"
    return True
