from app.entities.dubbing_job.services.recover_jobs.dtos.recover_jobs_dto_in import RecoverJobsDtoIn
from app.entities.dubbing_job.services.recover_jobs.dtos.recover_jobs_dto_out import RecoverJobsDtoOut
from app.interfaces.i_job_execution_repository import IJobExecutionRepository


class RecoverJobsService:
    def __init__(self, repository: IJobExecutionRepository):
        self.repository = repository

    def exec(self, dto: RecoverJobsDtoIn) -> RecoverJobsDtoOut:
        if dto.max_attempts < 1:
            raise ValueError("Invalid retry policy")
        return RecoverJobsDtoOut(self.repository.recover(dto.max_attempts))
