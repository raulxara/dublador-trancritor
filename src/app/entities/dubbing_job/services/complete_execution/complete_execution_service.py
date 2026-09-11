from app.entities.dubbing_job.services.complete_execution.dtos.complete_execution_dto_in import CompleteExecutionDtoIn
from app.entities.dubbing_job.services.complete_execution.dtos.complete_execution_dto_out import CompleteExecutionDtoOut
from app.interfaces.i_execution_results_repository import IExecutionResultsRepository


class CompleteExecutionService:
    def __init__(self, repository: IExecutionResultsRepository):
        self.repository = repository

    def exec(self, dto: CompleteExecutionDtoIn) -> CompleteExecutionDtoOut:
        return CompleteExecutionDtoOut(self.repository.complete(dto.lease, dto.result))
