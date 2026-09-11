from dataclasses import dataclass

from app.entities.dubbing_job.job_lease_entity import JobLeaseEntity


@dataclass(frozen=True, slots=True)
class RenewJobDtoIn:
    lease: JobLeaseEntity
    lease_seconds: int = 120
