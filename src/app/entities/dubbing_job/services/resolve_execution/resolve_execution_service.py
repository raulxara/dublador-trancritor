from app.entities.dubbing_job.services.resolve_execution.dtos.resolve_execution_dto_in import ResolveExecutionDtoIn
from app.entities.dubbing_job.services.resolve_execution.dtos.resolve_execution_dto_out import ResolveExecutionDtoOut
from app.interfaces.i_execution_results_repository import IExecutionResultsRepository


class ResolveExecutionService:
    def __init__(self, repository: IExecutionResultsRepository):
        self.repository = repository

    def exec(self, dto: ResolveExecutionDtoIn) -> ResolveExecutionDtoOut:
        return ResolveExecutionDtoOut(self.repository.resolve(dto.lease))
