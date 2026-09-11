from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequireOwnedJobDtoIn:
    office_id: str
    owner_id: str
    job_id: str
