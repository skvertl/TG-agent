from typing import Optional, List
from core.observability.models import (
    GlobalStats,
    RunSummary,
    RunTimeline,
)


def format_token_count(n: int) -> str:
    """Форматирует число токенов в компактный вид (500, 1.5k, 4.2M)."""
    if n >= 1_000_000:
        val = n / 1_000_000.0
        return f"{val:.1f}M" if val < 100 else f"{int(val)}M"
    elif n >= 1_000:
        val = n / 1_000.0
        return f"{val:.1f}k" if val < 100 else f"{int(val)}k"
    return str(n)


def format_telegram_report(
    stats: GlobalStats,
    recent_runs: Optional[List[RunSummary]] = None,
    last_timeline: Optional[RunTimeline] = None,
    project_name: Optional[str] = None,
) -> str:
    """
    Форматирует отчет для Telegram в чистом Markdown.
    Включает:
    1. Глобальные метрики All-Time
    2. Историю последних диалогов
    3. Таймлайн последнего запуска
    """
    title_suffix = f" [{project_name}]" if project_name else ""
    lines = [
        f"📊 *AI AGENT OBSERVABILITY{title_suffix}*",
        "─" * 28,
        f"🎯 *Всего задач:* `{stats.total_tasks}`",
        "",
        "💰 *Токены и стоимость:*",
        f"  • Входные (Input): `{format_token_count(stats.total_input_tokens)}`",
        f"  • Выходные (Output): `{format_token_count(stats.total_output_tokens)}`",
        f"  • Кэшированные (Cached): `{format_token_count(stats.total_cached_tokens)}`",
        f"  • Итоговая стоимость: *${stats.total_cost:.4f}*",
        "",
        "📈 *Эффективность:*",
        f"  • В среднем на задачу: `{format_token_count(int(stats.avg_tokens_per_task))}` токенов",
        f"  • Среднее число ходов: `{stats.avg_turns_per_task:.1f}`",
        f"  • Cache Hit Rate: `{stats.cache_hit_rate:.1f}%`",
        f"  • Повторный контекст: `{stats.repeated_tokens_pct:.1f}%`",
    ]

    if stats.tool_usage_pct:
        lines.append("")
        lines.append("🛠 *Топ инструментов (по токенам):*")
        for tool, pct in list(stats.tool_usage_pct.items())[:4]:
            bar = "█" * max(1, int(pct / 10))
            lines.append(f"  • `{tool:<12}` {bar} `{pct:.1f}%`")

    if recent_runs:
        lines.append("")
        lines.append("─" * 28)
        lines.append("📜 *История последних диалогов:*")
        for r in recent_runs[:5]:
            status_emoji = "✅" if r.success else "❌"
            cost_str = f"${r.total_cost:.4f}"
            tokens_str = format_token_count(r.total_input_tokens + r.total_output_tokens)
            # Короткий ID
            short_id = r.task_id if len(r.task_id) <= 12 else r.task_id[:12] + "…"
            lines.append(
                f"{status_emoji} `{short_id}` | `{tokens_str}` tok | `{cost_str}` | `{r.turns_count}h`"
            )

    if last_timeline and last_timeline.turns:
        lines.append("")
        lines.append("─" * 28)
        lines.append(f"⏱ *Таймлайн последнего запуска* (`{last_timeline.summary.task_id}`):")
        for turn in last_timeline.turns[:6]:
            llm_tok = format_token_count(turn.llm_span.input_tokens) if turn.llm_span else "0"
            lines.append(f"  *Turn {turn.turn_number}:* LLM `{llm_tok}` tok")
            for tool in turn.tool_spans:
                tool_tok = format_token_count(tool.output_tokens)
                lines.append(f"    └ ⚙️ `{tool.tool_name}`: `{tool_tok}` tok")

    return "\n".join(lines)


def format_timeline_markdown(timeline: RunTimeline) -> str:
    """Форматирует детальный таймлайн одного конкретного запуска."""
    s = timeline.summary
    total_tok = format_token_count(s.total_input_tokens + s.total_output_tokens)
    lines = [
        f"⏱ *Таймлайн задачи:* `{s.task_id}`",
        f"Модель: `{s.model}` | Всего токенов: `{total_tok}` | Стоимость: *${s.total_cost:.4f}*",
        "─" * 28,
    ]

    for turn in timeline.turns:
        llm_input = format_token_count(turn.llm_span.input_tokens) if turn.llm_span else "0"
        llm_out = format_token_count(turn.llm_span.output_tokens) if turn.llm_span else "0"
        lines.append(f"▶️ *Turn {turn.turn_number}*")
        lines.append(f"   • LLM: `{llm_input}` in / `{llm_out}` out")
        for tool in turn.tool_spans:
            tool_tok = format_token_count(tool.output_tokens)
            dur = f"{tool.duration_ms:.0f}ms"
            lines.append(f"   • ⚙️ Tool `{tool.tool_name}`: `{tool_tok}` tok ({dur})")

    return "\n".join(lines)


def render_cli_dashboard(
    stats: GlobalStats,
    recent_runs: Optional[List[RunSummary]] = None,
    last_timeline: Optional[RunTimeline] = None,
    project_name: Optional[str] = None,
) -> None:
    """Отрисовывает красивый интерактивный дашборд в терминале через Rich."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
        from rich import box
    except ImportError:
        # Fallback на текстовый вывод
        print(format_telegram_report(stats, recent_runs, last_timeline, project_name))
        return

    console = Console()

    # 1. Заголовок
    proj_text = f" [cyan]({project_name})[/cyan]" if project_name else ""
    console.print(Panel.fit(f"[bold green]AI AGENT OBSERVABILITY DASHBOARD[/bold green]{proj_text}", box=box.ROUNDED))

    # 2. Метрики (Таблица-сводка)
    summary_table = Table(title="Global Summary (All-Time)", box=box.SIMPLE_HEAVY)
    summary_table.add_column("Metric", style="cyan", justify="left")
    summary_table.add_column("Value", style="bold yellow", justify="right")

    summary_table.add_row("Tasks Completed", str(stats.total_tasks))
    summary_table.add_row("Total Input Tokens", format_token_count(stats.total_input_tokens))
    summary_table.add_row("Total Output Tokens", format_token_count(stats.total_output_tokens))
    summary_table.add_row("Total Cached Tokens", format_token_count(stats.total_cached_tokens))
    summary_table.add_row("Estimated Cost", f"${stats.total_cost:.4f}")
    summary_table.add_row("Avg Tokens / Task", format_token_count(int(stats.avg_tokens_per_task)))
    summary_table.add_row("Avg Turns / Task", f"{stats.avg_turns_per_task:.1f}")
    summary_table.add_row("Cache Hit Rate", f"{stats.cache_hit_rate:.1f}%")
    summary_table.add_row("Repeated Context", f"{stats.repeated_tokens_pct:.1f}%")

    # Инструменты
    tool_table = Table(title="Most Expensive Tools", box=box.SIMPLE_HEAVY)
    tool_table.add_column("Tool", style="magenta")
    tool_table.add_column("Share", style="bold green", justify="right")

    if stats.tool_usage_pct:
        for tool, pct in stats.tool_usage_pct.items():
            tool_table.add_row(tool, f"{pct:.1f}%")
    else:
        tool_table.add_row("No tool data", "-")

    console.print(Columns([summary_table, tool_table]))

    # 3. История последних диалогов
    if recent_runs:
        history_table = Table(title="Recent Dialogues / Runs History", box=box.ROUNDED)
        history_table.add_column("Task ID", style="cyan")
        history_table.add_column("Model", style="blue")
        history_table.add_column("Turns", justify="right")
        history_table.add_column("Tokens", justify="right", style="yellow")
        history_table.add_column("Repeated", justify="right", style="red")
        history_table.add_column("Cost ($)", justify="right", style="green")
        history_table.add_column("Status", justify="center")

        for r in recent_runs:
            status = "[green]SUCCESS[/green]" if r.success else "[red]FAILED[/red]"
            total_tok = format_token_count(r.total_input_tokens + r.total_output_tokens)
            rep_tok = format_token_count(r.repeated_tokens)
            history_table.add_row(
                r.task_id[:16],
                r.model,
                str(r.turns_count),
                total_tok,
                rep_tok,
                f"${r.total_cost:.4f}",
                status,
            )
        console.print(history_table)

    # 4. Таймлайн последнего запуска
    if last_timeline and last_timeline.turns:
        t_table = Table(
            title=f"Timeline for Task: [bold cyan]{last_timeline.summary.task_id}[/bold cyan] ({last_timeline.summary.model})",
            box=box.MINIMAL_DOUBLE_HEAD,
        )
        t_table.add_column("Turn", justify="center", style="bold")
        t_table.add_column("Component", style="magenta")
        t_table.add_column("Action / Tool", style="blue")
        t_table.add_column("Tokens", justify="right", style="yellow")
        t_table.add_column("Duration", justify="right", style="green")

        for turn in last_timeline.turns:
            if turn.llm_span:
                t_table.add_row(
                    f"Turn {turn.turn_number}",
                    "LLM Call",
                    turn.llm_span.model,
                    f"{format_token_count(turn.llm_span.input_tokens)} in",
                    f"{turn.llm_span.latency_ms:.0f}ms",
                )
            for tool in turn.tool_spans:
                t_table.add_row(
                    f"Turn {turn.turn_number}",
                    "Tool Execution",
                    tool.tool_name,
                    f"{format_token_count(tool.output_tokens)} out",
                    f"{tool.duration_ms:.0f}ms",
                )
        console.print(t_table)
