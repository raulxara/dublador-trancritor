from dataclasses import dataclass

from sqlalchemy import Engine

from app.entities.dubbing_job.services.recover_jobs.recover_jobs_service import RecoverJobsService
from app.models.processing.sqlalchemy_job_execution_repository import SqlAlchemyJobExecutionRepository
from app.providers.database_engine_factory import DatabaseEngineFactory
from app.use_cases.recover_expired_jobs.recover_expired_jobs_use_case_service import RecoverExpiredJobsUseCaseService


@dataclass(slots=True)
class SchedulerContainer:
    engine: Engine
    recover: RecoverExpiredJobsUseCaseService

    @classmethod
    def build(cls, settings):
        engine = DatabaseEngineFactory.build(settings)
        return cls(
            engine, RecoverExpiredJobsUseCaseService(RecoverJobsService(SqlAlchemyJobExecutionRepository(engine)))
        )
