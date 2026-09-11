from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListDubbingMessagesDtoOut:
    data: list[dict]
