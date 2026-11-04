"""Неизменяемые результаты экспериментов, отдельно от старых таблиц"""

import json
import hashlib
import base64
import zlib
from pathlib import Path
import platform
from importlib.metadata import distributions
from uuid import uuid4

import pandas as pd
from psycopg2.extras import Json, execute_values

from portfolio_app.db import get_connection

CREATE_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS mart.experiment_runs (
    run_id UUID PRIMARY KEY,
    experiment TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    config JSONB NOT NULL
)
"""
CREATE_RESULTS_SQL = """
CREATE TABLE IF NOT EXISTS mart.experiment_results (
    run_id UUID NOT NULL REFERENCES mart.experiment_runs(run_id),
    dataset TEXT NOT NULL,
    row_number BIGINT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (run_id, dataset, row_number)
)
"""
CREATE_CODE_SQL = """
CREATE TABLE IF NOT EXISTS mart.code_versions (
    code_version TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    files JSONB NOT NULL,
    encoding TEXT NOT NULL DEFAULT 'utf8'
);
ALTER TABLE mart.code_versions ADD COLUMN IF NOT EXISTS encoding TEXT NOT NULL DEFAULT 'utf8'
"""


def frame_batches(frame, batch_size=1000):
    """Преобразует по одному пакету: даты в строки ISO, NaN в null."""
    for start in range(0, len(frame), batch_size):
        records = json.loads(
            frame.iloc[start : start + batch_size].to_json(
                orient="records",
                date_format="iso",
                double_precision=15,
            )
        )
        yield start, records


def save_experiment(experiment, frames, config, connection=None):
    """Сохраняет полный запуск одной транзакцией и возвращает его уникальный идентификатор."""
    run_id = str(uuid4())
    root = Path(__file__).resolve().parents[1]
    code_paths = [
        path
        for pattern in [
            "*.py",
            "portfolio_app/*.py",
            "data_pipeline/*.py",
            "research/*.py",
            "tools/*.py",
            "tests/*.py",
            "requirements*.txt",
        ]
        for path in root.glob(pattern)
    ]
    code_bytes = {
        path.relative_to(root).as_posix(): path.read_bytes() for path in code_paths
    }
    code_hashes = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in code_bytes.items()
    }
    code_version = hashlib.sha256(
        json.dumps(code_hashes, sort_keys=True).encode()
    ).hexdigest()
    code_files = {
        name: base64.b64encode(zlib.compress(content)).decode("ascii")
        for name, content in code_bytes.items()
    }
    installed = {
        dist.metadata["Name"]: dist.version
        for dist in distributions()
        if dist.metadata.get("Name")
    }
    metadata = {
        "parameters": config,
        "python": platform.python_version(),
        "code_sha256": code_hashes,
        "code_version": code_version,
        "dependencies": installed,
    }
    owns_connection = connection is None
    conn = get_connection() if owns_connection else connection
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS mart")
            cur.execute(CREATE_RUNS_SQL)
            cur.execute(CREATE_RESULTS_SQL)
            cur.execute(CREATE_CODE_SQL)
            cur.execute(
                "INSERT INTO mart.code_versions(code_version,files,encoding) VALUES(%s,%s,'zlib+base64') ON CONFLICT DO NOTHING",
                (code_version, Json(code_files)),
            )
            cur.execute(
                "INSERT INTO mart.experiment_runs (run_id, experiment, config) VALUES (%s, %s, %s)",
                (run_id, experiment, Json(metadata)),
            )
            for dataset, frame in frames.items():
                for start, records in frame_batches(frame):
                    rows = [
                        (run_id, dataset, start + i, Json(row))
                        for i, row in enumerate(records)
                    ]
                    execute_values(
                        cur,
                        """
                        INSERT INTO mart.experiment_results (run_id, dataset, row_number, payload)
                        VALUES %s
                    """,
                        rows,
                        page_size=1000,
                    )
        if owns_connection:
            conn.commit()
    except Exception:
        if owns_connection:
            conn.rollback()
        raise
    finally:
        if owns_connection:
            conn.close()
    return run_id


def metrics_frame(metrics):
    return (
        pd.DataFrame.from_dict(metrics, orient="index")
        .rename_axis("portfolio_name")
        .reset_index()
    )
