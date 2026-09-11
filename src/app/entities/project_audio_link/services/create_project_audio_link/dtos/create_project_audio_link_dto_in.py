from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CreateProjectAudioLinkDtoIn:
    office_id: str
    user_id: str
    owner_id: str
    project_id: str
    output_id: str
