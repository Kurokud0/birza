"""Сервис БД для командной строки и будущего веб-интерфейса."""

from pathlib import Path
import hashlib
import base64
import zlib
from uuid import UUID, uuid4

from psycopg2.extras import Json, RealDictCursor

from portfolio_app.db import get_connection
from .config import PortfolioConfig

DATA_ROOT = Path(__file__).resolve().parents[1] / "data_snapshots"
SCHEMA = """
CREATE SCHEMA IF NOT EXISTS app;
CREATE TABLE IF NOT EXISTS app.portfolios (
    portfolio_id UUID PRIMARY KEY, name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS app.portfolio_versions (
    version_id UUID PRIMARY KEY, portfolio_id UUID NOT NULL REFERENCES app.portfolios,
    config JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS app.jobs (
    job_id UUID PRIMARY KEY, version_id UUID NOT NULL REFERENCES app.portfolio_versions,
    snapshot_id UUID NOT NULL, status TEXT NOT NULL DEFAULT 'queued'
        CHECK(status IN ('queued','running','succeeded','failed','cancelled')),
    progress DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK(progress BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ, heartbeat TIMESTAMPTZ, worker_id UUID,
    run_id UUID, error TEXT
);
ALTER TABLE app.jobs ADD COLUMN IF NOT EXISTS manifest_sha256 TEXT;
CREATE INDEX IF NOT EXISTS jobs_queue_idx ON app.jobs(created_at) WHERE status='queued';
CREATE INDEX IF NOT EXISTS jobs_version_idx ON app.jobs(version_id);
CREATE INDEX IF NOT EXISTS portfolio_versions_portfolio_idx ON app.portfolio_versions(portfolio_id,created_at);
CREATE TABLE IF NOT EXISTS app.job_logs (
    log_id BIGSERIAL PRIMARY KEY, job_id UUID NOT NULL REFERENCES app.jobs,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), level TEXT NOT NULL, message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS job_logs_job_idx ON app.job_logs(job_id,log_id);
"""


def transaction(action):
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            result = action(cur)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize():
    from portfolio_app.store import CREATE_RUNS_SQL, CREATE_RESULTS_SQL, CREATE_CODE_SQL
    from .input_store import SCHEMA as INPUT_SCHEMA

    def action(cur):
        cur.execute("CREATE SCHEMA IF NOT EXISTS mart")
        cur.execute(CREATE_RUNS_SQL)
        cur.execute(CREATE_RESULTS_SQL)
        cur.execute(CREATE_CODE_SQL)
        cur.execute(SCHEMA)
        cur.execute(INPUT_SCHEMA)
        cur.execute("""DO $$ BEGIN
            IF NOT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='app.jobs'::regclass AND conname='jobs_input_snapshot_fk') THEN
                ALTER TABLE app.jobs ADD CONSTRAINT jobs_input_snapshot_fk FOREIGN KEY(snapshot_id) REFERENCES app.input_snapshots;
            END IF;
            IF NOT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='app.jobs'::regclass AND conname='jobs_result_fk') THEN
                ALTER TABLE app.jobs ADD CONSTRAINT jobs_result_fk FOREIGN KEY(run_id) REFERENCES mart.experiment_runs;
            END IF;
        END $$""")

    transaction(action)


def create_portfolio(name, config):
    if not isinstance(config, PortfolioConfig) or not name.strip() or len(name) > 200:
        raise ValueError("Ожидаются корректная конфигурация и название портфеля")
    portfolio_id, version_id = str(uuid4()), str(uuid4())

    def action(cur):
        cur.execute(
            "INSERT INTO app.portfolios(portfolio_id,name) VALUES(%s,%s)",
            (portfolio_id, name.strip()),
        )
        cur.execute(
            "INSERT INTO app.portfolio_versions(version_id,portfolio_id,config) VALUES(%s,%s,%s)",
            (version_id, portfolio_id, Json(config.to_dict())),
        )

    transaction(action)
    return {"portfolio_id": portfolio_id, "version_id": version_id}


def add_version(portfolio_id, config):
    if not isinstance(config, PortfolioConfig):
        raise ValueError("Ожидается проверенная конфигурация PortfolioConfig")
    version = str(uuid4())
    transaction(
        lambda cur: cur.execute(
            "INSERT INTO app.portfolio_versions(version_id,portfolio_id,config) VALUES(%s,%s,%s)",
            (version, str(UUID(portfolio_id)), Json(config.to_dict())),
        )
    )
    return version


def snapshot_path(snapshot_id):
    return DATA_ROOT / str(UUID(snapshot_id))


def enqueue(version_id, snapshot_id):
    manifest = snapshot_path(snapshot_id) / "manifest.json"
    if manifest.is_file():
        from .input_store import register_snapshot

        register_snapshot(snapshot_path(snapshot_id))
        manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    else:
        from .input_store import database_manifest_hash

        manifest_hash = database_manifest_hash(snapshot_id)
    job_id = str(uuid4())
    transaction(
        lambda cur: cur.execute(
            "INSERT INTO app.jobs(job_id,version_id,snapshot_id,manifest_sha256) VALUES(%s,%s,%s,%s)",
            (job_id, str(UUID(version_id)), str(UUID(snapshot_id)), manifest_hash),
        )
    )
    return job_id


def get_job(job_id):
    def action(cur):
        cur.execute("SELECT * FROM app.jobs WHERE job_id=%s", (str(UUID(job_id)),))
        row = cur.fetchone()
        if row is None:
            raise ValueError("Задание не найдено")
        return dict(row)

    return transaction(action)


def list_portfolios(limit=100):
    def action(cur):
        cur.execute(
            "SELECT * FROM app.portfolios ORDER BY created_at DESC LIMIT %s",
            (min(max(int(limit), 1), 1000),),
        )
        return [dict(row) for row in cur.fetchall()]

    return transaction(action)


def list_instruments():
    def action(cur):
        cur.execute("""SELECT i.ticker,i.instrument_name,min(p.trading_date) AS history_start,
                              max(p.trading_date) AS history_end,count(p.trading_date) AS observations
                       FROM core.instruments i LEFT JOIN core.market_prices p USING(instrument_id)
                       GROUP BY i.instrument_id,i.ticker,i.instrument_name ORDER BY i.ticker""")
        return [dict(row) for row in cur.fetchall()]

    return transaction(action)


def read_results(run_id, dataset="daily", offset=0, limit=1000):
    if offset < 0 or not 1 <= limit <= 1000:
        raise ValueError("Некорректные параметры постраничного чтения")

    def action(cur):
        cur.execute(
            """SELECT payload FROM mart.experiment_results
                       WHERE run_id=%s AND dataset=%s ORDER BY row_number LIMIT %s OFFSET %s""",
            (str(UUID(run_id)), dataset, limit, offset),
        )
        return [row["payload"] for row in cur.fetchall()]

    return transaction(action)


def get_run(run_id):
    def action(cur):
        cur.execute(
            "SELECT * FROM mart.experiment_runs WHERE run_id=%s", (str(UUID(run_id)),)
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError("Расчёт не найден")
        return dict(row)

    return transaction(action)


def get_code_version(code_version):
    if len(code_version) != 64 or any(
        char not in "0123456789abcdef" for char in code_version
    ):
        raise ValueError("Некорректная версия кода")

    def action(cur):
        cur.execute(
            "SELECT files,encoding FROM mart.code_versions WHERE code_version=%s",
            (code_version,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError("Версия кода не найдена")
        if row["encoding"] == "utf8":
            return row["files"]
        if row["encoding"] != "zlib+base64":
            raise ValueError("Неподдерживаемый формат архива исходников")
        return {
            name: zlib.decompress(base64.b64decode(content)).decode("utf-8")
            for name, content in row["files"].items()
        }

    return transaction(action)


def read_logs(job_id, after=0, limit=100):
    if after < 0 or not 1 <= limit <= 1000:
        raise ValueError("Некорректные параметры постраничного чтения журнала")

    def action(cur):
        cur.execute(
            "SELECT * FROM app.job_logs WHERE job_id=%s AND log_id>%s ORDER BY log_id LIMIT %s",
            (str(UUID(job_id)), after, limit),
        )
        return [dict(row) for row in cur.fetchall()]

    return transaction(action)


def cancel_queued(job_id):
    def action(cur):
        cur.execute(
            "UPDATE app.jobs SET status='cancelled',finished_at=now() WHERE job_id=%s AND status='queued' RETURNING job_id",
            (str(UUID(job_id)),),
        )
        return cur.fetchone() is not None

    return transaction(action)


def get_version(version_id):
    def action(cur):
        cur.execute(
            "SELECT * FROM app.portfolio_versions WHERE version_id=%s",
            (str(UUID(version_id)),),
        )
        result = cur.fetchone()
        if result is None:
            raise ValueError("Версия портфеля не найдена")
        return dict(result)

    return transaction(action)


def list_jobs(limit=100):
    def action(cur):
        cur.execute(
            "SELECT * FROM app.jobs ORDER BY created_at DESC LIMIT %s",
            (min(max(int(limit), 1), 1000),),
        )
        return [dict(row) for row in cur.fetchall()]

    return transaction(action)


def fail_stale_jobs(seconds=3600):
    """Ручное восстановление после подтверждения остановки работника без автоматического повторного расчёта."""
    if seconds < 300:
        raise ValueError(
            "Период ожидания перед восстановлением должен быть не менее пяти минут"
        )

    def action(cur):
        cur.execute(
            "UPDATE app.jobs SET status='failed',finished_at=now(),error='Работник давно не обновлял отметку активности; требуется ручная проверка'\n                       WHERE status='running' AND heartbeat < now() - %s * interval '1 second' RETURNING job_id",
            (seconds,),
        )
        ids = [str(row["job_id"]) for row in cur.fetchall()]
        for job_id in ids:
            cur.execute(
                "INSERT INTO app.job_logs(job_id,level,message) VALUES(%s,'ERROR','Оператор отметил задание прекратившего работу исполнителя как неуспешное')",
                (job_id,),
            )
        return ids

    return transaction(action)
