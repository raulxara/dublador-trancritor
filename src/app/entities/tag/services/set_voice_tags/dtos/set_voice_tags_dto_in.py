from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SetVoiceTagsDtoIn:
    office_id: str
    user_id: str
    owner_id: str
    voice_id: str
    tag_ids: tuple[str, ...]
