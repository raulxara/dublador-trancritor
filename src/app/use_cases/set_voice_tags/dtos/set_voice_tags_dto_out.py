from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SetVoiceTagsDtoOut:
    data: list[dict]
