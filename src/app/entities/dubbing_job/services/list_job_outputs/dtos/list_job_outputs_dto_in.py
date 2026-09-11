from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListJobOutputsDtoIn:
    office_id: str
    job_id: str
