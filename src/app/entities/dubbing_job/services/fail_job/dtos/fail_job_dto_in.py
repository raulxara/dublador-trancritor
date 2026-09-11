from dataclasses import dataclass

from app.entities.dubbing_job.job_lease_entity import JobLeaseEntity


@dataclass(frozen=True, slots=True)
class FailJobDtoIn:
    lease: JobLeaseEntity
    code: str
