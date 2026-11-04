"""Копии неизменяемых входных снимков в БД. CSV служат переносимым кэшем."""

import hashlib
from pathlib import Path
from uuid import UUID

import pandas as pd
from psycopg2.extras import Json, execute_values

from portfolio_app.store import frame_batches
from .data import MarketData, load_snapshot
from .service import transaction

SCHEMA = """
CREATE TABLE IF NOT EXISTS app.input_snapshots (
    snapshot_id UUID PRIMARY KEY, manifest_sha256 TEXT NOT NULL, metadata JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS app.input_rows (
    snapshot_id UUID NOT NULL REFERENCES app.input_snapshots(snapshot_id),
    dataset TEXT NOT NULL,row_number BIGINT NOT NULL,payload JSONB NOT NULL,
    PRIMARY KEY(snapshot_id,dataset,row_number)
);
"""


def register_snapshot(directory):
    path = Path(directory)
    data = load_snapshot(path)
    snapshot_id = str(UUID(path.name))
    manifest_hash = hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
    frames = {
        "prices": data.prices,
        "rates": data.rates,
        "calendar": pd.DataFrame({"trading_date": data.calendar}),
        "dividends": data.dividends,
    }
    if data.fx is not None:
        frames["fx"] = data.fx

    def action(cur):
        cur.execute(SCHEMA)
        cur.execute(
            "SELECT manifest_sha256 FROM app.input_snapshots WHERE snapshot_id=%s FOR UPDATE",
            (snapshot_id,),
        )
        existing = cur.fetchone()
        if existing:
            if existing["manifest_sha256"] != manifest_hash:
                raise ValueError(
                    "Идентификатор снимка уже зарегистрирован с другим содержимым"
                )
            return snapshot_id
        cur.execute(
            "INSERT INTO app.input_snapshots(snapshot_id,manifest_sha256,metadata) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING RETURNING snapshot_id",
            (snapshot_id, manifest_hash, Json(data.metadata)),
        )
        if cur.fetchone() is None:
            cur.execute(
                "SELECT manifest_sha256 FROM app.input_snapshots WHERE snapshot_id=%s",
                (snapshot_id,),
            )
            if cur.fetchone()["manifest_sha256"] != manifest_hash:
                raise ValueError(
                    "При одновременной регистрации обнаружено другое содержимое снимка"
                )
            return snapshot_id
        for dataset, frame in frames.items():
            for start, records in frame_batches(frame):
                rows = [
                    (snapshot_id, dataset, start + i, Json(row))
                    for i, row in enumerate(records)
                ]
                execute_values(
                    cur,
                    "INSERT INTO app.input_rows(snapshot_id,dataset,row_number,payload) VALUES %s",
                    rows,
                    page_size=1000,
                )
        return snapshot_id

    return transaction(action)


def database_manifest_hash(snapshot_id):
    def action(cur):
        cur.execute(
            "SELECT manifest_sha256 FROM app.input_snapshots WHERE snapshot_id=%s",
            (str(UUID(snapshot_id)),),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError("Входной снимок не зарегистрирован")
        return row["manifest_sha256"]

    return transaction(action)


def load_database_snapshot(snapshot_id):
    def action(cur):
        cur.execute(
            "SELECT metadata FROM app.input_snapshots WHERE snapshot_id=%s",
            (str(UUID(snapshot_id)),),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError("Входной снимок не найден")
        metadata = row["metadata"]
        frames = {}
        for dataset in ["prices", "rates", "calendar", "dividends", "fx"]:
            cur.execute(
                "SELECT payload FROM app.input_rows WHERE snapshot_id=%s AND dataset=%s ORDER BY row_number",
                (snapshot_id, dataset),
            )
            frames[dataset] = pd.DataFrame([item["payload"] for item in cur.fetchall()])
        for name in ["prices", "rates", "calendar", "fx"]:
            if not frames[name].empty:
                frames[name]["trading_date"] = pd.to_datetime(frames[name].trading_date)
        if not frames["dividends"].empty:
            for name in ["ex_date", "payment_date"]:
                frames["dividends"][name] = pd.to_datetime(frames["dividends"][name])
        else:
            from .data import DIVIDEND_COLUMNS

            frames["dividends"] = pd.DataFrame(columns=DIVIDEND_COLUMNS)
        return MarketData(
            frames["prices"],
            frames["rates"],
            pd.DatetimeIndex(frames["calendar"].trading_date),
            frames["dividends"],
            metadata,
            frames["fx"] if not frames["fx"].empty else None,
        )

    return transaction(action)
