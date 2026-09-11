from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FailJobDtoOut:
    data: bool
