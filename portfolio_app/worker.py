"""Одно задание на запуск с атомарным получением. Внешний процесс может повторять вызов."""

from uuid import uuid4
import hashlib

from portfolio_app.store import save_experiment
from .config import PortfolioConfig
from .data import load_snapshot
from .engine import calculate
from .service import snapshot_path, transaction


def work_once():
    worker_id = str(uuid4())

    def claim(cur):
        cur.execute(
            """
            UPDATE app.jobs SET status='running',worker_id=%s,started_at=now(),heartbeat=now()
            WHERE job_id=(SELECT job_id FROM app.jobs WHERE status='queued'
                          ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
            RETURNING *
        """,
            (worker_id,),
        )
        job = cur.fetchone()
        if not job:
            return None
        cur.execute(
            "SELECT config FROM app.portfolio_versions WHERE version_id=%s",
            (job["version_id"],),
        )
        return dict(job), cur.fetchone()["config"]

    claimed = transaction(claim)
    if claimed is None:
        return None
    job, raw_config = claimed
    job_id = str(job["job_id"])

    def progress(fraction, message):
        def action(cur):
            cur.execute(
                "UPDATE app.jobs SET progress=%s,heartbeat=now() WHERE job_id=%s AND worker_id=%s AND status='running'",
                (fraction, job_id, worker_id),
            )
            if cur.rowcount != 1:
                raise RuntimeError("Работник больше не владеет этим заданием")
            cur.execute(
                "INSERT INTO app.job_logs(job_id,level,message) VALUES(%s,'INFO',%s)",
                (job_id, message),
            )

        transaction(action)

    try:
        progress(0, "Загрузка проверенного входного снимка")
        config = PortfolioConfig(**raw_config)
        path = snapshot_path(str(job["snapshot_id"]))
        if (path / "manifest.json").is_file():
            manifest_hash = hashlib.sha256(
                (path / "manifest.json").read_bytes()
            ).hexdigest()
            data = load_snapshot(path)
        else:
            from .input_store import database_manifest_hash, load_database_snapshot

            manifest_hash = database_manifest_hash(str(job["snapshot_id"]))
            data = load_database_snapshot(str(job["snapshot_id"]))
        if not job.get("manifest_sha256") or manifest_hash != job["manifest_sha256"]:
            raise ValueError(
                "Манифест входных данных изменён после постановки задания; создайте новое задание"
            )
        frames = calculate(config, data, progress)

        def finish(cur):
            cur.execute(
                "SELECT status,worker_id FROM app.jobs WHERE job_id=%s FOR UPDATE",
                (job_id,),
            )
            current = cur.fetchone()
            if (
                current is None
                or current["status"] != "running"
                or str(current["worker_id"]) != worker_id
            ):
                raise RuntimeError(
                    "Работник потерял право на задание до сохранения результата"
                )
            run_id = save_experiment(
                "application_portfolio",
                frames,
                {
                    "portfolio": config.to_dict(),
                    "snapshot": data.metadata,
                    "job_id": job_id,
                    "execution": "Сигнал по прошлому закрытию, исполнение на следующем открытии, дробные доли, односторонние издержки",
                },
                connection=cur.connection,
            )
            cur.execute(
                """UPDATE app.jobs SET status='succeeded',progress=1,run_id=%s,finished_at=now(),heartbeat=now()
                           WHERE job_id=%s AND worker_id=%s AND status='running'""",
                (run_id, job_id, worker_id),
            )
            cur.execute(
                "INSERT INTO app.job_logs(job_id,level,message) VALUES(%s,'INFO','Расчёт сохранён')",
                (job_id,),
            )

        transaction(finish)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:4000]

        def fail(cur):
            cur.execute(
                """UPDATE app.jobs SET status='failed',error=%s,finished_at=now(),heartbeat=now()
                           WHERE job_id=%s AND worker_id=%s AND status='running'""",
                (message, job_id, worker_id),
            )
            cur.execute(
                "INSERT INTO app.job_logs(job_id,level,message) VALUES(%s,'ERROR',%s)",
                (job_id, message),
            )

        transaction(fail)
    return job_id
