import importlib
import subprocess
from datetime import datetime

import questionary
from rich.console import Console
from rich.table import Table

console = Console()


def check_and_install(
    required: list[str],
    optional: list[dict],
    state: dict,
) -> tuple[bool, dict]:
    """
    Check required and optional packages before a stage runs.

    - Required packages: if any are missing, the stage cannot run.
      User is offered the option to install them.
    - Optional packages: if missing, user is informed and offered install.
      Declining skips the feature but does not block the stage.

    Each package entry in `optional` is a dict:
        {"package": "ydata-profiling", "reason": "HTML profiling reports"}

    Returns:
        (can_proceed, updated_state)
        can_proceed is False only if required packages are missing and user declines install.
    """
    missing_required = [pkg for pkg in required if not _is_installed(pkg)]
    missing_optional = [
        entry for entry in optional if not _is_installed(entry["package"])
    ]

    if missing_optional:
        _show_optional_table(missing_optional)
        for entry in missing_optional:
            approved = questionary.confirm(
                f"Install '{entry['package']}' to enable: {entry['reason']}?",
                default=False,
            ).ask()
            if approved:
                success = _install(entry["package"])
                state = _log(state, entry["package"], installed=success)
            else:
                state = _log(state, entry["package"], installed=False, skipped=True)

    if missing_required:
        _show_required_table(missing_required)
        approved = questionary.confirm(
            "Install missing required packages to continue?",
            default=True,
        ).ask()

        if not approved:
            console.print("[red]Required packages not installed. Cannot run this stage.[/red]")
            for pkg in missing_required:
                state = _log(state, pkg, installed=False, skipped=True)
            return False, state

        all_installed = True
        for pkg in missing_required:
            success = _install(pkg)
            state = _log(state, pkg, installed=success)
            if not success:
                all_installed = False

        if not all_installed:
            console.print("[red]Some required packages failed to install. Cannot run this stage.[/red]")
            return False, state

    return True, state


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _show_required_table(packages: list[str]) -> None:
    console.print("\n[bold red]Missing required packages:[/bold red]")
    table = Table(show_header=True, header_style="bold red")
    table.add_column("Package")
    for pkg in packages:
        table.add_row(pkg)
    console.print(table)


def _show_optional_table(packages: list[dict]) -> None:
    console.print("\n[bold yellow]Optional packages not installed:[/bold yellow]")
    table = Table(show_header=True, header_style="bold yellow")
    table.add_column("Package", style="cyan")
    table.add_column("Enables")
    for entry in packages:
        table.add_row(entry["package"], entry["reason"])
    console.print(table)


# ---------------------------------------------------------------------------
# Install and check
# ---------------------------------------------------------------------------

def _is_installed(package: str) -> bool:
    """Check if a package is importable. Handles package/import name differences."""
    import_name = _to_import_name(package)
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def _install(package: str) -> bool:
    """Install a package via uv into the current environment. Returns True on success."""
    console.print(f"[blue]Installing {package}...[/blue]")
    result = subprocess.run(
        ["uv", "pip", "install", package],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        console.print(f"[green]Successfully installed {package}.[/green]")
        return True
    else:
        console.print(f"[red]Failed to install {package}:[/red]\n{result.stderr}")
        return False


def _to_import_name(package: str) -> str:
    """
    Map pip package names to their import names where they differ.
    e.g. "scikit-learn" -> "sklearn", "pyyaml" -> "yaml"
    """
    mapping = {
        "scikit-learn": "sklearn",
        "pyyaml": "yaml",
        "ydata-profiling": "ydata_profiling",
        "pillow": "PIL",
    }
    return mapping.get(package.lower(), package.replace("-", "_"))


# ---------------------------------------------------------------------------
# State logging
# ---------------------------------------------------------------------------

def _log(
    state: dict,
    package: str,
    installed: bool,
    skipped: bool = False,
) -> dict:
    entry = {
        "package": package,
        "action": "skipped" if skipped else ("installed" if installed else "failed"),
        "at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if "dependencies" not in state:
        state["dependencies"] = []
    state["dependencies"].append(entry)
    return state
