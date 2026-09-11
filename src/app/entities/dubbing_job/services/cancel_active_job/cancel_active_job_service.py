from dataclasses import replace

from app.entities.dubbing_job.i_dubbing_job_repository import IJobRepository
from app.entities.dubbing_job.services.cancel_active_job.dtos.cancel_active_job_dto_in import CancelActiveJobDtoIn
from app.entities.dubbing_job.services.cancel_active_job.dtos.cancel_active_job_dto_out import CancelActiveJobDtoOut
from app.exceptions.conflict_error import ConflictError


class CancelActiveJobService:
    def __init__(self, repository: IJobRepository):
        self.repository = repository

    def exec(self, dto: CancelActiveJobDtoIn) -> CancelActiveJobDtoOut:
        if dto.job.processing_state not in ("queued", "processing", "cancelled"):
            raise ConflictError()
        self.repository.cancel(dto.job.office_id, dto.job.unique_id)
        return CancelActiveJobDtoOut(replace(dto.job, processing_state="cancelled"))
