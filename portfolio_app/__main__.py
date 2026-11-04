from portfolio_app.cli import RussianArgumentParser
import json
from pathlib import Path

from .config import PortfolioConfig
from . import service


def main():
    parser = RussianArgumentParser(
        description="Локальный расчёт портфеля без веб-сервера и обучения моделей"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db", help="Создать прикладные таблицы")
    snapshot = sub.add_parser("snapshot", help="Создать снимок входных данных")
    snapshot.add_argument("--reconcile-tcsg", action="store_true")
    divs = sub.add_parser(
        "import-dividends", help="Импортировать проверенные дивиденды"
    )
    divs.add_argument("snapshot_id")
    divs.add_argument("csv")
    divs.add_argument("coverage")
    fetch = sub.add_parser("fetch-dividends", help="Загрузить дивиденды для проверки")
    fetch.add_argument("--tickers", nargs="+", required=True)
    fetch.add_argument("--start", required=True)
    fetch.add_argument("--end", required=True)
    create = sub.add_parser("create", help="Создать портфель из настроек")
    create.add_argument("name")
    create.add_argument("config")
    queue = sub.add_parser("enqueue", help="Поставить расчёт в очередь")
    queue.add_argument("version_id")
    queue.add_argument("snapshot_id")
    sub.add_parser("work-once", help="Обработать одно задание")
    register = sub.add_parser("register-snapshot", help="Зарегистрировать снимок в БД")
    register.add_argument("snapshot_id")
    reconcile = sub.add_parser("reconcile-inputs", help="Сверить входы с источниками")
    reconcile.add_argument("snapshot_id")
    reconcile.add_argument("--sources", nargs="+", required=True)
    reconcile.add_argument("--reference", required=True)
    reconcile.add_argument("--calendar", required=True)
    job = sub.add_parser("job", help="Показать состояние задания")
    job.add_argument("job_id")
    logs = sub.add_parser("logs", help="Показать журнал задания")
    logs.add_argument("job_id")
    result = sub.add_parser("results", help="Прочитать результаты расчёта")
    result.add_argument("run_id")
    result.add_argument("--dataset", default="metrics")
    result.add_argument("--offset", type=int, default=0)
    result.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "init-db":
        service.initialize()
        output = {"status": "initialized"}
    elif args.command == "snapshot":
        from .data import read_market, save_snapshot

        data = read_market()
        if args.reconcile_tcsg:
            from .sources import reconcile_tcsg

            data = reconcile_tcsg(data)
        output = {"path": str(save_snapshot(data, service.DATA_ROOT))}
    elif args.command == "import-dividends":
        from .sources import import_dividends

        output = {
            "path": str(
                import_dividends(
                    service.snapshot_path(args.snapshot_id),
                    args.csv,
                    args.coverage,
                    service.DATA_ROOT,
                )
            )
        }
    elif args.command == "fetch-dividends":
        from .sources import download_tbank_dividends

        output = {
            "path": str(download_tbank_dividends(args.tickers, args.start, args.end))
        }
    elif args.command == "create":
        config = PortfolioConfig(
            **json.loads(Path(args.config).read_text(encoding="utf-8"))
        )
        output = service.create_portfolio(args.name, config)
    elif args.command == "enqueue":
        output = {"job_id": service.enqueue(args.version_id, args.snapshot_id)}
    elif args.command == "work-once":
        from .worker import work_once

        output = {"job_id": work_once()}
    elif args.command == "register-snapshot":
        from .input_store import register_snapshot

        output = {
            "snapshot_id": register_snapshot(service.snapshot_path(args.snapshot_id))
        }
    elif args.command == "reconcile-inputs":
        from .data import load_snapshot, read_market, save_snapshot
        from .reconciliation import verify_calendar, reconcile_dividends

        data = load_snapshot(service.snapshot_path(args.snapshot_id))
        data.fx = read_market().fx
        verify_calendar(data, args.calendar)
        reconcile_dividends(data, args.sources, args.reference)
        output = {"path": str(save_snapshot(data, service.DATA_ROOT))}
    elif args.command == "job":
        output = service.get_job(args.job_id)
    elif args.command == "logs":
        output = service.read_logs(args.job_id)
    else:
        output = service.read_results(
            args.run_id, args.dataset, args.offset, args.limit
        )
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
