from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListVoicesDtoOut:
    data: list[dict]
