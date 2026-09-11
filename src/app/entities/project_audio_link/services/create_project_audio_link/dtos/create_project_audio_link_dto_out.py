from dataclasses import dataclass

from app.entities.project_audio_link.project_audio_link_entity import ProjectAudioLinkEntity


@dataclass(frozen=True, slots=True)
class CreateProjectAudioLinkDtoOut:
    data: ProjectAudioLinkEntity
