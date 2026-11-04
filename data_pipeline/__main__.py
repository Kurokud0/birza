"""Явный запуск этапов работы с данными. Обучение вынесено в отдельную команду исследований."""

from portfolio_app.cli import RussianArgumentParser
from pathlib import Path
import subprocess
import sys
from time import perf_counter

WORK_DIR = Path(__file__).resolve().parents[1]
STEPS = {
    "etl": "data_pipeline.etl",
    "transform": "data_pipeline.transform",
    "quality": "data_pipeline.quality",
}
DEFAULT_STEPS = ["etl", "transform", "quality"]


def main(argv=None):
    parser = RussianArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        default=DEFAULT_STEPS,
        help="Этапы загрузки, преобразования и проверки",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Показать этапы без их выполнения"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Загрузить историю с 2020 года вместо продолжения с последней сохранённой даты",
    )
    args = parser.parse_args(argv)
    for name in args.steps:
        command = [sys.executable, "-m", STEPS[name]]
        if name == "etl" and args.full:
            command.append("--full")
        print(f"{name}: {command}", flush=True)
        if args.dry_run:
            continue
        started = perf_counter()
        result = subprocess.run(command, cwd=WORK_DIR)
        print(
            f"{name}: код завершения={result.returncode}, секунд={perf_counter() - started:.1f}",
            flush=True,
        )
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
