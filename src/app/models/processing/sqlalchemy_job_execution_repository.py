from sqlalchemy import Engine, text

from app.entities.dubbing_job.job_lease_entity import JobLeaseEntity


class SqlAlchemyJobExecutionRepository:
    """Short, independently committed queue transactions; never performs inference."""

    def __init__(self, engine: Engine):
        self.engine = engine

    @staticmethod
    def parameters(lease):
        return dict(id=lease.job_id, office=lease.office_id, token=lease.token)

    def claim(self, token, lease_seconds, max_attempts):
        with self.engine.begin() as connection:
            row = (
                connection.execute(
                    text("""SELECT _id,office_id,user_customer_id,attempts
                FROM dubbing_jobs WHERE status='active' AND processing_state='queued'
                AND attempts < :maximum ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED"""),
                    dict(maximum=max_attempts),
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            connection.execute(
                text("""UPDATE dubbing_jobs SET processing_state='processing',
                worker_token=:token, locked_until=TIMESTAMPADD(SECOND,:seconds,UTC_TIMESTAMP()),
                attempts=attempts+1, started_at=UTC_TIMESTAMP(), finished_at=NULL,
                error_code=NULL,error_message=NULL,updated_at=UTC_TIMESTAMP() WHERE _id=:id"""),
                dict(id=row["_id"], token=token, seconds=lease_seconds),
            )
            return JobLeaseEntity(row["_id"], row["office_id"], row["user_customer_id"], token, row["attempts"] + 1)

    def renew(self, lease, lease_seconds):
        with self.engine.begin() as connection:
            return (
                connection.execute(
                    text("""UPDATE dubbing_jobs
                SET locked_until=TIMESTAMPADD(SECOND,:seconds,UTC_TIMESTAMP()),updated_at=UTC_TIMESTAMP()
                WHERE _id=:id AND office_id=:office AND worker_token=:token
                AND status='active' AND processing_state='processing' AND locked_until>UTC_TIMESTAMP()"""),
                    dict(self.parameters(lease), seconds=lease_seconds),
                ).rowcount
                == 1
            )

    def fail(self, lease, code):
        with self.engine.begin() as connection:
            return (
                connection.execute(
                    text("""UPDATE dubbing_jobs SET processing_state='failed',
                error_code=:code,error_message=NULL,finished_at=UTC_TIMESTAMP(),updated_at=UTC_TIMESTAMP(),
                worker_token=NULL,locked_until=NULL WHERE _id=:id AND office_id=:office
                AND worker_token=:token AND status='active' AND processing_state='processing'
                AND locked_until>UTC_TIMESTAMP()"""),
                    dict(self.parameters(lease), code=code),
                ).rowcount
                == 1
            )

    def recover(self, max_attempts):
        with self.engine.begin() as connection:
            rows = (
                connection.execute(
                    text("""SELECT _id FROM dubbing_jobs
                WHERE processing_state='processing' AND (locked_until IS NULL OR locked_until<=UTC_TIMESTAMP())
                ORDER BY id LIMIT 100 FOR UPDATE SKIP LOCKED""")
                )
                .scalars()
                .all()
            )
            for identifier in rows:
                connection.execute(
                    text("""UPDATE dubbing_jobs SET
                    processing_state=IF(attempts>=:maximum OR status='inactive','failed','queued'),
                    finished_at=IF(attempts>=:maximum OR status='inactive',UTC_TIMESTAMP(),NULL),
                    error_code='LEASE_EXPIRED',error_message=NULL,worker_token=NULL,locked_until=NULL,
                    updated_at=UTC_TIMESTAMP() WHERE _id=:id"""),
                    dict(id=identifier, maximum=max_attempts),
                )
            return len(rows)
