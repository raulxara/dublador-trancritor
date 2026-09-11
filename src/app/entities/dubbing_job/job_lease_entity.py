from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class JobLeaseEntity:
    job_id: str
    office_id: str
    owner_id: str
    token: str = field(repr=False)
    attempt: int = 1
