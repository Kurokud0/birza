"""Подключение к PostgreSQL: переменные окружения имеют приоритет над локальными настройками."""

import json
import os
from pathlib import Path
import psycopg2


def get_connection():
    path = Path(__file__).resolve().parents[1] / "local_db.json"
    settings = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    names = {
        "PGHOST": "host",
        "PGPORT": "port",
        "PGDATABASE": "database",
        "PGUSER": "user",
        "PGPASSWORD": "password",
    }
    for variable, parameter in names.items():
        if variable in os.environ:
            settings[parameter] = os.environ[variable]
    if not settings:
        raise RuntimeError(
            "Настройте PostgreSQL через переменные PG* или файл local_db.json"
        )
    settings.setdefault("connect_timeout", 10)
    return psycopg2.connect(**settings)


if __name__ == "__main__":
    connection = get_connection()
    connection.close()
    print("Подключение к PostgreSQL: успешно")
