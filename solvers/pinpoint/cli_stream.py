"""Interactive Pinpoint stream: type clues one by one and watch the ranking sharpen."""

from __future__ import annotations

from collections.abc import Callable

from rich.console import Console
from rich.table import Table

from solvers.pinpoint.semantic_engine import CategoryGuess, SemanticEngine

SHOWN = 5


def ranking_table(clue_number: int, ranking: list[CategoryGuess]) -> Table:
    table = Table(title=f"After clue {clue_number}", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("Category")
    table.add_column("Confidence", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Cosine", justify="right")
    for rank, guess in enumerate(ranking[:SHOWN], start=1):
        table.add_row(
            str(rank), guess.label, f"{guess.confidence:5.1f}%", f"{guess.coverage:.0%}", f"{guess.cosine:.2f}"
        )
    return table


def run_stream(
    engine: SemanticEngine, read_clue: Callable[[str], str] = input, console: Console | None = None
) -> list[CategoryGuess]:
    """Prompt for clues until a blank line, ``quit`` or EOF; return the final ranking."""
    console = console or Console()
    console.print("[bold]Pinpoint stream[/bold] — enter clues one at a time (blank line or 'quit' to finish).")
    ranking: list[CategoryGuess] = []
    while True:
        try:
            clue = read_clue(f"Clue {len(engine.clues) + 1}: ").strip()
        except EOFError:
            break
        if not clue or clue.lower() == "quit":
            break
        ranking = engine.add_clue(clue)
        if ranking:
            console.print(ranking_table(len(engine.clues), ranking))
        else:
            console.print(f"[yellow]'{clue}' has no WordNet noun sense; centroid updated only.[/yellow]")
    return ranking
