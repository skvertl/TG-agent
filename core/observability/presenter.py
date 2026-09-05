from typing import Optional, List
from core.observability.models import (
    GlobalStats,
    RunSummary,
    RunTimeline,
)
from core.observability.pricing import get_model_rates, get_pricing_explanation


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
    1. Глобальные метрики All-Time и используемую модель
    2. Расчет стоимости и формулу
    3. Историю последних диалогов
    4. Таймлайн последнего запуска
    """
    title_suffix = f" [{project_name}]" if project_name else ""
    active_model = (last_timeline.summary.model if last_timeline else stats.active_model) or "qwen2.5:7b"
    rates = get_model_rates(active_model)
    in_rate = rates["input_per_1m"]
    out_rate = rates["output_per_1m"]
    cache_rate = rates.get("cached_per_1m", in_rate * 0.25)

    lines = [
        f"📊 *AI AGENT OBSERVABILITY{title_suffix}*",
        "─" * 28,
        f"🤖 *Используемая модель:* `{active_model}`",
        f"🎯 *Всего задач:* `{stats.total_tasks}`",
        "",
        "💰 *Токены и стоимость:*",
        f"  • Входные (Input): `{format_token_count(stats.total_input_tokens)}`",
        f"  • Выходные (Output): `{format_token_count(stats.total_output_tokens)}`",
        f"  • Кэшированные (Cached): `{format_token_count(stats.total_cached_tokens)}`",
        f"  • Итоговая стоимость: *${stats.total_cost:.4f}*",
        "",
        "💡 *Как рассчитывается стоимость:*",
        f"  • Тарифы: `${in_rate:.2f}`/1M вх. | `${out_rate:.2f}`/1M вых. | `${cache_rate:.3f}`/1M кэш",
        f"  • Формула: `(Input×${in_rate} + Output×${out_rate} + Cache×${cache_rate}) / 1M`",
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
            short_id = r.task_id if len(r.task_id) <= 12 else r.task_id[:12] + "…"
            lines.append(
                f"{status_emoji} `{short_id}` ({r.model}) | `{tokens_str}` tok | `{cost_str}` | `{r.turns_count}h`"
            )

    if last_timeline and last_timeline.turns:
        s = last_timeline.summary
        s_rates = get_model_rates(s.model)
        calc_formula = f"({s.total_input_tokens}*${s_rates['input_per_1m']} + {s.total_output_tokens}*${s_rates['output_per_1m']}) / 1M"
        lines.append("")
        lines.append("─" * 28)
        lines.append(f"⏱ *Таймлайн последнего запуска* (`{s.task_id}`):")
        lines.append(f"🤖 Модель: `{s.model}` | Расчет: `{calc_formula} = ${s.total_cost:.4f}`")
        for turn in last_timeline.turns[:6]:
            llm_tok = format_token_count(turn.llm_span.input_tokens) if turn.llm_span else "0"
            lines.append(f"  *Turn {turn.turn_number}:* LLM `{llm_tok}` tok")
            for tool in turn.tool_spans:
                tool_tok = format_token_count(tool.output_tokens)
                lines.append(f"    └ ⚙️ `{tool.tool_name}`: `{tool_tok}` tok")

    return "\n".join(lines)


def format_timeline_markdown(timeline: RunTimeline) -> str:
    """Форматирует детальный таймлайн одного конкретного запуска с формулой цены."""
    s = timeline.summary
    total_tok = format_token_count(s.total_input_tokens + s.total_output_tokens)
    rates = get_model_rates(s.model)
    calc_formula = f"({s.total_input_tokens} * ${rates['input_per_1m']} + {s.total_output_tokens} * ${rates['output_per_1m']}) / 1M"

    lines = [
        f"⏱ *Таймлайн задачи:* `{s.task_id}`",
        f"🤖 *Модель:* `{s.model}`",
        f"💰 *Всего токенов:* `{total_tok}` (In: {s.total_input_tokens} / Out: {s.total_output_tokens} / Cache: {s.total_cached_tokens})",
        f"💡 *Расчет стоимости:* `{calc_formula} = ${s.total_cost:.4f}*`",
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

    active_model = (last_timeline.summary.model if last_timeline else stats.active_model) or "qwen2.5:7b"
    rates = get_model_rates(active_model)

    # 2. Метрики (Таблица-сводка)
    summary_table = Table(title="Global Summary (All-Time)", box=box.SIMPLE_HEAVY)
    summary_table.add_column("Metric", style="cyan", justify="left")
    summary_table.add_column("Value", style="bold yellow", justify="right")

    summary_table.add_row("Active Model", f"[bold green]{active_model}[/bold green]")
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

    # Детализация тарифов и формулы расчета
    calc_panel = Panel(
        f"[bold]Active Model:[/bold] [cyan]{active_model}[/cyan]\n"
        f"[bold]Rates per 1M tokens:[/bold] Input: [yellow]${rates['input_per_1m']:.2f}[/yellow] | "
        f"Output: [yellow]${rates['output_per_1m']:.2f}[/yellow] | "
        f"Cache: [yellow]${rates.get('cached_per_1m', 0.0):.3f}[/yellow]\n"
        f"[bold]Cost Formula:[/bold] [green]Cost = (Input*${rates['input_per_1m']} + Output*${rates['output_per_1m']} + Cache*${rates.get('cached_per_1m', 0.0):.3f}) / 1,000,000[/green]",
        title="Cost Calculation & Pricing Details",
        box=box.ROUNDED,
    )
    console.print(calc_panel)

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
