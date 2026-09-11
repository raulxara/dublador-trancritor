from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CreateTagDtoOut:
    data: dict
