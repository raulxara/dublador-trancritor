from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListVoiceTagsDtoOut:
    data: list[dict]
