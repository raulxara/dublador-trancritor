from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UnlinkProjectAudioDtoIn:
    office_id: str
    user_id: str
    owner_id: str
    link_id: str
