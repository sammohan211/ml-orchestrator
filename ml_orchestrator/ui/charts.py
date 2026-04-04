import plotext as plt
from rich.console import Console

console = Console()

CHART_WIDTH = 70


def horizontal_bar(labels: list[str], values: list[float], title: str, unit: str = "") -> None:
    """Render a horizontal bar chart inline in the terminal."""
    if not labels or not values:
        console.print(f"[dim]{title}: no data to display.[/dim]")
        return

    plt.clear_figure()
    plt.bar(labels, values, orientation="h")
    plt.title(title)
    plt.plot_size(CHART_WIDTH, min(len(labels) + 4, 30))
    if unit:
        plt.xlabel(unit)
    plt.show()


def bar(labels: list[str], values: list[float], title: str, unit: str = "") -> None:
    """Render a vertical bar chart inline in the terminal."""
    if not labels or not values:
        console.print(f"[dim]{title}: no data to display.[/dim]")
        return

    plt.clear_figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.plot_size(CHART_WIDTH, 20)
    if unit:
        plt.ylabel(unit)
    plt.show()
