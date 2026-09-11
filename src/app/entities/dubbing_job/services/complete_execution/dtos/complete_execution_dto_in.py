from dataclasses import dataclass

from app.entities.dubbing_job.execution_result_entity import ExecutionResultEntity
from app.entities.dubbing_job.job_lease_entity import JobLeaseEntity


@dataclass(frozen=True, slots=True)
class CompleteExecutionDtoIn:
    lease: JobLeaseEntity
    result: ExecutionResultEntity
