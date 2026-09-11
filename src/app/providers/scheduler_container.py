from dataclasses import dataclass

from sqlalchemy import Engine

from app.entities.dubbing_job.services.recover_jobs.recover_jobs_service import RecoverJobsService
from app.models.media.sqlalchemy_media_inventory_repository import SqlAlchemyMediaInventoryRepository
from app.models.processing.sqlalchemy_job_execution_repository import SqlAlchemyJobExecutionRepository
from app.providers.database_engine_factory import DatabaseEngineFactory
from app.services.media.orphan_cleanup_service import OrphanCleanupService
from app.use_cases.cleanup_orphan_media.cleanup_orphan_media_use_case_service import CleanupOrphanMediaUseCaseService
from app.use_cases.recover_expired_jobs.recover_expired_jobs_use_case_service import RecoverExpiredJobsUseCaseService


@dataclass(slots=True)
class SchedulerContainer:
    cleanup: CleanupOrphanMediaUseCaseService
    engine: Engine
    recover: RecoverExpiredJobsUseCaseService

    @classmethod
    def build(cls, settings):
        engine = DatabaseEngineFactory.build(settings)
        return cls(
            CleanupOrphanMediaUseCaseService(
                OrphanCleanupService(settings.media_root, SqlAlchemyMediaInventoryRepository(engine))
            ),
            engine,
            RecoverExpiredJobsUseCaseService(RecoverJobsService(SqlAlchemyJobExecutionRepository(engine))),
        )
