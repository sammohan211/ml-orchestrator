import typer
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ml_orchestrator.state.manager import create_project, load_project
from ml_orchestrator.stages import ingestion, profiling, cleaning, feature_preparation, feature_selection, modeling

app = typer.Typer(add_completion=False)
console = Console()

MENU_ITEMS = [
    ("1", "Start New Project"),
    ("2", "Load Existing Project"),
    ("3", "Read Dataset"),
    ("4", "Profile Dataset"),
    ("5", "Clean Dataset"),
    ("6", "Prepare Features"),
    ("7", "Select Features"),
    ("8", "Train Baseline Model"),
    ("9", "Tune Hyperparameters"),
    ("10", "Evaluate Model"),
    ("11", "Export Artifacts"),
    ("12", "Settings"),
    ("13", "Exit"),
]


def show_menu(state: dict | None) -> None:
    lines = Text()
    for key, label in MENU_ITEMS:
        status = _stage_status(key, state)
        lines.append(f"  {key:>2}.  {label}  {status}\n")
    title = f"ML Workflow Helper — {state['project']['name']}" if state else "ML Workflow Helper"
    console.print(Panel(lines, title=title, border_style="blue"))


def _stage_status(menu_key: str, state: dict | None) -> str:
    """Return a coloured status indicator for menu items that map to a stage."""
    if state is None:
        return ""
    stage_map = {
        "3": "ingestion",
        "4": "profiling",
        "5": "cleaning",
        "6": "feature_preparation",
        "7": "feature_selection",
        "8": "modeling",
        "9": "tuning",
        "10": "evaluation",
        "11": "export",
    }
    stage = stage_map.get(menu_key)
    if not stage:
        return ""
    status = state.get("stages", {}).get(stage, "pending")
    if status == "completed":
        return "[green]✓[/green]"
    if status == "skipped":
        return "[dim]—[/dim]"
    return ""


def handle_choice(choice: str, state: dict | None) -> tuple[bool, dict | None]:
    """
    Dispatch menu selection.
    Returns (should_continue, updated_state).
    """
    match choice:
        case "1":
            state = _start_new_project()
        case "2":
            state = _load_existing_project()
        case "3":
            if state is None:
                console.print("[yellow]No project loaded. Please create or load a project first (options 1 or 2).[/yellow]")
            else:
                state = ingestion.run(state)
        case "4":
            if state is None:
                console.print("[yellow]No project loaded. Please create or load a project first (options 1 or 2).[/yellow]")
            else:
                state = profiling.run(state)
        case "5":
            if state is None:
                console.print("[yellow]No project loaded. Please create or load a project first (options 1 or 2).[/yellow]")
            else:
                state = cleaning.run(state)
        case "6":
            if state is None:
                console.print("[yellow]No project loaded. Please create or load a project first (options 1 or 2).[/yellow]")
            else:
                state = feature_preparation.run(state)
        case "7":
            if state is None:
                console.print("[yellow]No project loaded. Please create or load a project first (options 1 or 2).[/yellow]")
            else:
                state = feature_selection.run(state)
        case "8":
            if state is None:
                console.print("[yellow]No project loaded. Please create or load a project first (options 1 or 2).[/yellow]")
            else:
                state = modeling.run(state)
        case "9":
            console.print("[yellow]Tune Hyperparameters — not yet implemented.[/yellow]")
        case "10":
            console.print("[yellow]Evaluate Model — not yet implemented.[/yellow]")
        case "11":
            console.print("[yellow]Export Artifacts — not yet implemented.[/yellow]")
        case "12":
            console.print("[yellow]Settings — not yet implemented.[/yellow]")
        case "13":
            console.print("[blue]Goodbye.[/blue]")
            return False, state
        case _:
            console.print("[red]Invalid selection. Please enter a number from 1 to 13.[/red]")
    return True, state


def _start_new_project() -> dict | None:
    from pathlib import Path

    name = questionary.text(
        "Project name:",
        validate=lambda v: bool(v.strip()) or "Project name cannot be empty.",
    ).ask()
    if name is None:
        return None

    directory = questionary.text(
        "Parent directory for this project:",
        default=str(Path.home()),
        validate=lambda v: Path(v).expanduser().exists() or "Directory does not exist.",
    ).ask()
    if directory is None:
        return None

    try:
        return create_project(name.strip(), str(Path(directory.strip()).expanduser()))
    except Exception as e:
        console.print(f"[red]Failed to create project: {e}[/red]")
        return None


def _load_existing_project() -> dict | None:
    from pathlib import Path

    path = questionary.text(
        "Path to project directory or state.yaml:",
        validate=lambda v: Path(v).expanduser().exists() or "Path does not exist.",
    ).ask()
    if path is None:
        return None

    try:
        return load_project(str(Path(path.strip()).expanduser()))
    except Exception as e:
        console.print(f"[red]Failed to load project: {e}[/red]")
        return None


@app.command()
def main() -> None:
    """ML Workflow Helper — terminal-based ML orchestrator for tabular data."""
    valid_choices = {key for key, _ in MENU_ITEMS}
    state: dict | None = None

    while True:
        console.print()
        show_menu(state)

        choice = questionary.text(
            "Select an option:",
            validate=lambda val: val in valid_choices or "Enter a number from 1 to 13.",
        ).ask()

        if choice is None:
            console.print("\n[blue]Goodbye.[/blue]")
            break

        console.print()
        should_continue, state = handle_choice(choice, state)
        if not should_continue:
            break
