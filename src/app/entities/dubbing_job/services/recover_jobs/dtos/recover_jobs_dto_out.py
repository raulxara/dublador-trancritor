from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecoverJobsDtoOut:
    data: int
