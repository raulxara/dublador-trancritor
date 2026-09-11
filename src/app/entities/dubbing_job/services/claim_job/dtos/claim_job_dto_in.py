from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClaimJobDtoIn:
    lease_seconds: int = 120
    max_attempts: int = 3
