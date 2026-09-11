from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecoverJobsDtoIn:
    max_attempts: int = 3
