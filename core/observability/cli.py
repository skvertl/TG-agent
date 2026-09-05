import argparse
import sys
from core.observability.storage import TelemetryStorage
from core.observability.presenter import render_cli_dashboard, format_timeline_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Agent Observability CLI Dashboard"
    )
    parser.add_argument(
        "--project", "-p",
        type=str,
        default=None,
        help="Filter stats by project name (e.g. tg-agent)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite telemetry database",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=5,
        help="Number of recent runs to show",
    )
    parser.add_argument(
        "--run-id", "-r",
        type=str,
        default=None,
        help="Display detailed timeline for a specific task ID",
    )

    args = parser.parse_args()

    storage = TelemetryStorage(db_path=args.db)

    # Если запрошен таймлайн конкретной задачи
    if args.run_id:
        timeline = storage.get_run_timeline(args.run_id)
        if not timeline:
            print(f"Task with ID '{args.run_id}' not found in telemetry store.")
            sys.exit(1)
        # Отрисовываем таймлайн через presenter
        render_cli_dashboard(
            stats=storage.get_global_stats(project_name=args.project),
            recent_runs=None,
            last_timeline=timeline,
            project_name=args.project,
        )
        return

    # Стандартный дашборд
    stats = storage.get_global_stats(project_name=args.project)
    recent = storage.get_recent_runs(limit=args.limit, project_name=args.project)
    last_timeline = storage.get_run_timeline(recent[0].task_id) if recent else None

    render_cli_dashboard(
        stats=stats,
        recent_runs=recent,
        last_timeline=last_timeline,
        project_name=args.project,
    )


if __name__ == "__main__":
    main()
