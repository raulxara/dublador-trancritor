from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FindIdempotentJobDtoIn:
    office_id: str
    owner_id: str
    key: str
    request_hash: str
