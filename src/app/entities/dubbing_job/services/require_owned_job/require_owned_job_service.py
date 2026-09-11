from app.entities.dubbing_job.i_dubbing_job_repository import IJobRepository
from app.entities.dubbing_job.services.require_owned_job.dtos.require_owned_job_dto_in import RequireOwnedJobDtoIn
from app.entities.dubbing_job.services.require_owned_job.dtos.require_owned_job_dto_out import RequireOwnedJobDtoOut
from app.exceptions.resource_not_found_error import ResourceNotFoundError


class RequireOwnedJobService:
    def __init__(self, repository: IJobRepository):
        self.repository = repository

    def exec(self, dto: RequireOwnedJobDtoIn) -> RequireOwnedJobDtoOut:
        job = self.repository.find(dto.office_id, dto.job_id)
        if job is None or job.user_customer_id != dto.owner_id or job.status != "active":
            raise ResourceNotFoundError()
        return RequireOwnedJobDtoOut(job)
