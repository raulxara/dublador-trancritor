from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GetDubbingJobDtoOut:
    data: dict
