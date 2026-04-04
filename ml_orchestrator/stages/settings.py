import questionary
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ml_orchestrator.config.defaults import SETTINGS_PROMPTS
from ml_orchestrator.state.manager import save_state

console = Console()

# Map key -> prompt metadata for quick lookup
_PROMPT_MAP = {p["key"]: p for p in SETTINGS_PROMPTS}


def run(state: dict) -> dict:
    console.print("\n[bold blue]── Project Settings ──[/bold blue]\n")

    if not state.get("settings"):
        console.print("[yellow]No project loaded. Please create or load a project first (options 1 or 2).[/yellow]")
        return state

    while True:
        _show_settings(state)

        choices = [p["label"] for p in SETTINGS_PROMPTS] + ["Done — return to menu"]
        action = questionary.select("Edit a setting or return to menu:", choices=choices).ask()
        if action is None or action == "Done — return to menu":
            break

        # Find matching prompt
        prompt = next((p for p in SETTINGS_PROMPTS if p["label"] == action), None)
        if prompt is None:
            break

        key = prompt["key"]
        current = state["settings"].get(key)
        new_value = _prompt_value(prompt, current)

        if new_value is not None:
            state["settings"][key] = new_value
            save_state(state)
            console.print(f"  [green]'{key}' updated to {new_value}[/green]\n")

    console.print("[green]Settings saved.[/green]\n")
    return state


def _show_settings(state: dict) -> None:
    settings = state.get("settings", {})

    console.print(Panel(
        f"Project: [bold]{state['project']['name']}[/bold]",
        border_style="cyan",
    ))

    table = Table(box=box.SIMPLE, header_style="bold blue")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    table.add_column("Description", style="dim")

    for p in SETTINGS_PROMPTS:
        key = p["key"]
        value = settings.get(key, "—")
        table.add_row(key, str(value), p["label"])

    console.print(table)


def _prompt_value(prompt: dict, current):
    value_type = prompt["type"]
    label = prompt["label"]

    raw = questionary.text(
        f"{label}:",
        default=str(current) if current is not None else "",
        validate=lambda v: _validate(v, value_type),
    ).ask()

    if raw is None:
        return None

    return _cast(raw.strip(), value_type)


def _validate(value: str, value_type) -> bool | str:
    v = value.strip()
    if value_type is bool:
        if v.lower() in ("true", "false", "yes", "no", "1", "0"):
            return True
        return "Enter true or false."
    try:
        value_type(v)
        return True
    except ValueError:
        return f"Enter a valid {value_type.__name__}."


def _cast(value: str, value_type):
    if value_type is bool:
        return value.lower() in ("true", "yes", "1")
    return value_type(value)
