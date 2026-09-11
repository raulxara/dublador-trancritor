from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListTranscriptionsDtoOut:
    data: list[dict]
