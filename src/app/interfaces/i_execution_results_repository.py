from typing import Protocol

from app.entities.dubbing_job.execution_input_entity import ExecutionInputEntity
from app.entities.dubbing_job.execution_result_entity import ExecutionResultEntity
from app.entities.dubbing_job.job_lease_entity import JobLeaseEntity


class IExecutionResultsRepository(Protocol):
    def resolve(self, lease: JobLeaseEntity) -> ExecutionInputEntity | None: ...
    def complete(self, lease: JobLeaseEntity, result: ExecutionResultEntity) -> bool: ...
