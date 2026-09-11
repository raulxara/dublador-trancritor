from secrets import token_hex

from app.entities.dubbing_job.services.claim_job.dtos.claim_job_dto_in import ClaimJobDtoIn
from app.entities.dubbing_job.services.claim_job.dtos.claim_job_dto_out import ClaimJobDtoOut
from app.interfaces.i_job_execution_repository import IJobExecutionRepository


class ClaimJobService:
    def __init__(self, repository: IJobExecutionRepository):
        self.repository = repository

    def exec(self, dto: ClaimJobDtoIn) -> ClaimJobDtoOut:
        if dto.lease_seconds < 10 or dto.max_attempts < 1:
            raise ValueError("Invalid lease policy")
        return ClaimJobDtoOut(self.repository.claim(token_hex(32), dto.lease_seconds, dto.max_attempts))
