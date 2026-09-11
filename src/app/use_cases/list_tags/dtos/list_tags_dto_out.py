from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListTagsDtoOut:
    data: list[dict]
