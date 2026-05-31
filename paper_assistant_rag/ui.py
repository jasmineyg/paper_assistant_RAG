from __future__ import annotations

import sys

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


console = Console()


def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def safe_for_console(text: str) -> str:
    # Windows 终端有时不是 UTF-8，PDF 里又常有特殊字符；这里避免打印时报编码错误。
    encoding = console.file.encoding or sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def print_sources(source_rows: list[dict[str, str]], show_snippets: bool) -> None:
    # 把检索到的片段来源打印出来，这是 RAG 比普通聊天更可信的关键。
    table = Table(title="Retrieved Sources")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Paper")
    table.add_column("Page", justify="right")
    table.add_column("Chunk", justify="right")
    table.add_column("Score", justify="right")
    for row in source_rows:
        table.add_row(
            row["id"],
            safe_for_console(row["source"]),
            row["page"],
            row["chunk"],
            row["score"],
        )
    console.print(table)

    if show_snippets:
        console.print("\n[bold]Source snippets[/bold]")
        for row in source_rows:
            console.print(
                safe_for_console(
                    f"\n[cyan][{row['id']}][/cyan] {row['source']} | page {row['page']} | chunk {row['chunk']}"
                )
            )
            console.print(safe_for_console(row["snippet"]))

