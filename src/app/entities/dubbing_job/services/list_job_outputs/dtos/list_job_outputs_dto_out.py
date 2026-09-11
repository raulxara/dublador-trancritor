from dataclasses import dataclass

from app.entities.dubbing_job.job_output_entity import JobOutputEntity


@dataclass(frozen=True, slots=True)
class ListJobOutputsDtoOut:
    data: tuple[JobOutputEntity, ...]
