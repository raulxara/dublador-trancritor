from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProjectAudioLinkEntity:
    unique_id: str
    office_id: str
    external_project_id: str
    media_file_id: str
    created_by_user_id: str
    status: str = "active"
