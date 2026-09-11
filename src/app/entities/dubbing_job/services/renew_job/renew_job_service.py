from app.entities.dubbing_job.services.renew_job.dtos.renew_job_dto_in import RenewJobDtoIn
from app.entities.dubbing_job.services.renew_job.dtos.renew_job_dto_out import RenewJobDtoOut
from app.interfaces.i_job_execution_repository import IJobExecutionRepository


class RenewJobService:
    def __init__(self, repository: IJobExecutionRepository):
        self.repository = repository

    def exec(self, dto: RenewJobDtoIn) -> RenewJobDtoOut:
        if dto.lease_seconds < 10:
            raise ValueError("Invalid lease policy")
        return RenewJobDtoOut(self.repository.renew(dto.lease, dto.lease_seconds))
