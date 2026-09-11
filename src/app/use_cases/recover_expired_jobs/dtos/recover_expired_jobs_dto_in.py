from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecoverExpiredJobsDtoIn:
    max_attempts: int = 3
