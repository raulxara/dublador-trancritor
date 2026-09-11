from dataclasses import dataclass

from sqlalchemy import Engine

from app.entities.dubbing_job.services.claim_job.claim_job_service import ClaimJobService
from app.entities.dubbing_job.services.complete_execution.complete_execution_service import CompleteExecutionService
from app.entities.dubbing_job.services.fail_job.fail_job_service import FailJobService
from app.entities.dubbing_job.services.recover_jobs.recover_jobs_service import RecoverJobsService
from app.entities.dubbing_job.services.renew_job.renew_job_service import RenewJobService
from app.entities.dubbing_job.services.resolve_execution.resolve_execution_service import ResolveExecutionService
from app.models.processing.sqlalchemy_execution_results_repository import SqlAlchemyExecutionResultsRepository
from app.models.processing.sqlalchemy_job_execution_repository import SqlAlchemyJobExecutionRepository
from app.providers.database_engine_factory import DatabaseEngineFactory
from app.services.processing.subprocess_audio_execution import SubprocessAudioExecution
from app.use_cases.process_next_job.process_next_job_use_case_service import ProcessNextJobUseCaseService
from app.use_cases.recover_expired_jobs.recover_expired_jobs_use_case_service import RecoverExpiredJobsUseCaseService


@dataclass(slots=True)
class WorkerContainer:
    engine: Engine
    process: ProcessNextJobUseCaseService
    recover: RecoverExpiredJobsUseCaseService
    audio: SubprocessAudioExecution

    @classmethod
    def build(cls, settings, policy, health_callback=lambda: None, stopping=lambda: False):
        engine = DatabaseEngineFactory.build(settings)
        queue = SqlAlchemyJobExecutionRepository(engine)
        results = SqlAlchemyExecutionResultsRepository(engine)
        audio = SubprocessAudioExecution(settings.media_root, policy, health_callback, stopping)
        return cls(
            engine,
            ProcessNextJobUseCaseService(
                ClaimJobService(queue),
                RenewJobService(queue),
                FailJobService(queue),
                ResolveExecutionService(results),
                CompleteExecutionService(results),
                audio,
            ),
            RecoverExpiredJobsUseCaseService(RecoverJobsService(queue)),
            audio,
        )
