from app.entities.dubbing_job.services.fail_job.dtos.fail_job_dto_in import FailJobDtoIn
from app.entities.dubbing_job.services.fail_job.dtos.fail_job_dto_out import FailJobDtoOut
from app.interfaces.i_job_execution_repository import IJobExecutionRepository


class FailJobService:
    def __init__(self, repository: IJobExecutionRepository):
        self.repository = repository

    def exec(self, dto: FailJobDtoIn) -> FailJobDtoOut:
        if dto.code not in {"ENGINE_FAILED", "INPUT_UNAVAILABLE", "EXECUTION_TIMEOUT"}:
            raise ValueError("Unknown failure code")
        return FailJobDtoOut(self.repository.fail(dto.lease, dto.code))
