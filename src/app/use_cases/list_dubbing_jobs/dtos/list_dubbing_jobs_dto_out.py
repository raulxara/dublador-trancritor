from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListDubbingJobsDtoOut:
    data: list[dict]
