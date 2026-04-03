import typer
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

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


def show_menu() -> None:
    lines = Text()
    for key, label in MENU_ITEMS:
        lines.append(f"  {key:>2}.  {label}\n")
    console.print(Panel(lines, title="ML Workflow Helper", border_style="blue"))


def handle_choice(choice: str) -> bool:
    """
    Dispatch menu selection. Returns False when the user chooses to exit,
    True otherwise so the menu loop continues.
    """
    match choice:
        case "1":
            console.print("[yellow]Start New Project — not yet implemented.[/yellow]")
        case "2":
            console.print("[yellow]Load Existing Project — not yet implemented.[/yellow]")
        case "3":
            console.print("[yellow]Read Dataset — not yet implemented.[/yellow]")
        case "4":
            console.print("[yellow]Profile Dataset — not yet implemented.[/yellow]")
        case "5":
            console.print("[yellow]Clean Dataset — not yet implemented.[/yellow]")
        case "6":
            console.print("[yellow]Prepare Features — not yet implemented.[/yellow]")
        case "7":
            console.print("[yellow]Select Features — not yet implemented.[/yellow]")
        case "8":
            console.print("[yellow]Train Baseline Model — not yet implemented.[/yellow]")
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
            return False
        case _:
            console.print("[red]Invalid selection. Please enter a number from 1 to 13.[/red]")
    return True


@app.command()
def main() -> None:
    """ML Workflow Helper — terminal-based ML orchestrator for tabular data."""
    valid_choices = {key for key, _ in MENU_ITEMS}

    while True:
        console.print()
        show_menu()

        choice = questionary.text(
            "Select an option:",
            validate=lambda val: val in valid_choices or "Enter a number from 1 to 13.",
        ).ask()

        if choice is None:
            # User pressed Ctrl+C at the prompt
            console.print("\n[blue]Goodbye.[/blue]")
            break

        console.print()
        should_continue = handle_choice(choice)
        if not should_continue:
            break
