from app.entities.dubbing_job.services.list_job_outputs.dtos.list_job_outputs_dto_in import ListJobOutputsDtoIn
from app.entities.dubbing_job.services.list_job_outputs.dtos.list_job_outputs_dto_out import ListJobOutputsDtoOut
from app.interfaces.i_job_outputs_repository import IJobOutputsRepository


class ListJobOutputsService:
    def __init__(self, repository: IJobOutputsRepository):
        self.repository = repository

    def exec(self, dto: ListJobOutputsDtoIn) -> ListJobOutputsDtoOut:
        return ListJobOutputsDtoOut(tuple(self.repository.list(dto.office_id, dto.job_id)))
