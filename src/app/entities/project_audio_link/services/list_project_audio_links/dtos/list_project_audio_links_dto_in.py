from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListProjectAudioLinksDtoIn:
    office_id: str
    user_id: str
    owner_id: str
    project_id: str
    limit: int
    offset: int
