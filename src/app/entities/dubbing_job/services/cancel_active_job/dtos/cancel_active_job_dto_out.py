from dataclasses import dataclass

from app.entities.dubbing_job.dubbing_job_entity import DubbingJobEntity


@dataclass(frozen=True, slots=True)
class CancelActiveJobDtoOut:
    data: DubbingJobEntity
