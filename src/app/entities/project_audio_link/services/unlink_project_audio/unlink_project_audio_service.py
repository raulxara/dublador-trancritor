from dataclasses import replace

from app.entities.project_audio_link.i_project_audio_links_repository import IProjectAudioLinksRepository
from app.entities.project_audio_link.services.unlink_project_audio.dtos.unlink_project_audio_dto_in import (
    UnlinkProjectAudioDtoIn,
)
from app.entities.project_audio_link.services.unlink_project_audio.dtos.unlink_project_audio_dto_out import (
    UnlinkProjectAudioDtoOut,
)
from app.exceptions.resource_not_found_error import ResourceNotFoundError


class UnlinkProjectAudioService:
    def __init__(self, repository: IProjectAudioLinksRepository):
        self.repository = repository

    def exec(self, dto: UnlinkProjectAudioDtoIn) -> UnlinkProjectAudioDtoOut:
        entity = self.repository.find_id(dto.office_id, dto.link_id)
        if entity is None or entity.created_by_user_id != dto.user_id:
            raise ResourceNotFoundError()
        entity = replace(entity, status="inactive")
        self.repository.save(entity, False)
        return UnlinkProjectAudioDtoOut(entity)
