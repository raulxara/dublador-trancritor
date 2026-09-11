from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListDubbingOutputsDtoOut:
    data: list[dict]
