from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProcessNextJobDtoIn:
    lease_seconds: int = 120
    max_attempts: int = 3
