from app.entities.dubbing_job.services.recover_jobs.dtos.recover_jobs_dto_in import RecoverJobsDtoIn
from app.entities.dubbing_job.services.recover_jobs.recover_jobs_service import RecoverJobsService
from app.use_cases.recover_expired_jobs.dtos.recover_expired_jobs_dto_in import RecoverExpiredJobsDtoIn
from app.use_cases.recover_expired_jobs.dtos.recover_expired_jobs_dto_out import RecoverExpiredJobsDtoOut


class RecoverExpiredJobsUseCaseService:
    def __init__(self, service: RecoverJobsService):
        self.service = service

    def exec(self, dto: RecoverExpiredJobsDtoIn) -> RecoverExpiredJobsDtoOut:
        return RecoverExpiredJobsDtoOut(self.service.exec(RecoverJobsDtoIn(dto.max_attempts)).data)
