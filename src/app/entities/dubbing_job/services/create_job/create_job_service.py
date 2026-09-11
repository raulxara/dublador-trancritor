from app.entities.dubbing_job.i_dubbing_job_repository import IJobRepository
from app.entities.dubbing_job.services.create_job.dtos.create_job_dto_in import CreateJobDtoIn
from app.entities.dubbing_job.services.create_job.dtos.create_job_dto_out import CreateJobDtoOut


class CreateJobService:
    def __init__(self, repository: IJobRepository):
        self.repository = repository

    def exec(self, dto: CreateJobDtoIn) -> CreateJobDtoOut:
        self.repository.create(dto.job)
        return CreateJobDtoOut(dto.job)
