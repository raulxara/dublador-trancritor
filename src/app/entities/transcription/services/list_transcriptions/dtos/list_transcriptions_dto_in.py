from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListTranscriptionsDtoIn:
    office_id: str
    user_id: str
    owner_id: str
    job_id: str
    limit: int
    offset: int
