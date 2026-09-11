from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListVoiceTagsDtoIn:
    office_id: str
    user_id: str
    owner_id: str
    voice_id: str
