import os
import tempfile
from datetime import datetime
from pathlib import Path

import questionary
import yaml
from rich.console import Console
from rich.table import Table

from ml_orchestrator.config.defaults import DEFAULT_SETTINGS, SETTINGS_PROMPTS

console = Console()

SCHEMA_VERSION = "1.0"

STAGE_KEYS = [
    "ingestion",
    "profiling",
    "cleaning",
    "feature_preparation",
    "feature_selection",
    "modeling",
    "tuning",
    "evaluation",
    "export",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_project(name: str, directory: str) -> dict:
    """
    Create a new project: prompt user to confirm or change settings,
    build initial state, write state.yaml, return state dict.
    """
    project_dir = Path(directory) / name
    artifacts_dir = project_dir / "artifacts"
    project_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)

    console.print(f"\n[green]Project directory created:[/green] {project_dir}\n")

    settings = _prompt_settings()

    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    state = {
        "project": {
            "name": name,
            "project_dir": str(project_dir),
            "created_at": now,
            "last_updated": now,
            "schema_version": SCHEMA_VERSION,
        },
        "settings": settings,
        "stages": {key: "pending" for key in STAGE_KEYS},
        "ingestion": None,
        "profiling": None,
        "cleaning": None,
        "feature_preparation": None,
        "feature_selection": None,
        "modeling": None,
        "tuning": None,
        "evaluation": None,
        "artifacts": {},
        "dependencies": [],
    }

    save_state(state)
    console.print(f"[green]Project '{name}' created and state saved.[/green]\n")
    return state


def load_project(path: str) -> dict:
    """
    Load an existing project from a state.yaml file or project directory.
    Returns the state dict.
    """
    state_path = _resolve_state_path(path)

    if not state_path.exists():
        raise FileNotFoundError(f"No state.yaml found at: {state_path}")

    with open(state_path, "r", encoding="utf-8") as f:
        state = yaml.safe_load(f)

    if not isinstance(state, dict):
        raise ValueError(f"state.yaml at {state_path} is corrupt or empty.")

    stored_version = state.get("project", {}).get("schema_version")
    if stored_version != SCHEMA_VERSION:
        console.print(
            f"[yellow]Warning: state schema version '{stored_version}' "
            f"differs from current '{SCHEMA_VERSION}'. Some fields may be missing.[/yellow]"
        )

    console.print(
        f"[green]Project '{state['project']['name']}' loaded.[/green] "
        f"Last updated: {state['project']['last_updated']}\n"
    )
    return state


def save_state(state: dict) -> None:
    """
    Write state to disk using a temp-file-then-rename strategy to prevent
    corruption if the process is interrupted mid-write.
    """
    state_path = _resolve_state_path(state["project"]["project_dir"])
    state["project"]["last_updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    dir_path = state_path.parent
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=dir_path,
        delete=False,
        suffix=".tmp",
        encoding="utf-8",
    ) as tmp:
        yaml.dump(state, tmp, default_flow_style=False, allow_unicode=True, sort_keys=False)
        tmp_path = tmp.name

    os.replace(tmp_path, state_path)


def update_state(state: dict, **kwargs) -> dict:
    """
    Merge kwargs into state dict. Does not write to disk.
    Caller is responsible for calling save_state() after update.
    """
    state.update(kwargs)
    return state


# ---------------------------------------------------------------------------
# Settings prompt
# ---------------------------------------------------------------------------

def _prompt_settings() -> dict:
    """
    Show current defaults in a table, then ask the user to accept or change
    each value. Returns the confirmed settings dict.
    """
    console.print("[bold]Project Settings[/bold]\n")
    console.print("Review the default settings below. You can change any value or press Enter to accept.\n")

    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Setting", style="cyan")
    table.add_column("Default", justify="right")

    for prompt in SETTINGS_PROMPTS:
        table.add_row(prompt["label"], str(DEFAULT_SETTINGS[prompt["key"]]))

    console.print(table)
    console.print()

    settings = {}
    for prompt in SETTINGS_PROMPTS:
        key = prompt["key"]
        default = DEFAULT_SETTINGS[key]
        value_type = prompt["type"]

        raw = questionary.text(
            f"{prompt['label']}",
            default=str(default),
            validate=lambda val, t=value_type: _validate_type(val, t),
        ).ask()

        if raw is None:
            # Ctrl+C during settings — fall back to default
            settings[key] = default
        else:
            settings[key] = _cast(raw, value_type)

    return settings


def _validate_type(value: str, value_type: type) -> bool | str:
    try:
        if value_type == bool:
            if value.lower() not in ("true", "false"):
                return "Enter 'true' or 'false'."
        else:
            value_type(value)
        return True
    except ValueError:
        return f"Expected a {value_type.__name__} value."


def _cast(value: str, value_type: type):
    if value_type == bool:
        return value.lower() == "true"
    return value_type(value)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_state_path(path: str | Path) -> Path:
    """
    Accept either a project directory or a direct path to state.yaml.
    Always returns a Path pointing to state.yaml.
    """
    p = Path(path)
    if p.is_file():
        return p
    return p / "state.yaml"
