from app.entities.dubbing_job.i_dubbing_job_repository import IJobRepository
from app.entities.dubbing_job.services.find_idempotent_job.dtos.find_idempotent_job_dto_in import FindIdempotentJobDtoIn
from app.entities.dubbing_job.services.find_idempotent_job.dtos.find_idempotent_job_dto_out import (
    FindIdempotentJobDtoOut,
)
from app.exceptions.conflict_error import ConflictError


class FindIdempotentJobService:
    def __init__(self, repository: IJobRepository):
        self.repository = repository

    def exec(self, dto: FindIdempotentJobDtoIn) -> FindIdempotentJobDtoOut:
        self.repository.lock_owner(dto.office_id, dto.owner_id)
        job = self.repository.by_key(dto.office_id, dto.owner_id, dto.key)
        if job is not None and (job.request_hash != dto.request_hash or job.status != "active"):
            raise ConflictError()
        return FindIdempotentJobDtoOut(job)
