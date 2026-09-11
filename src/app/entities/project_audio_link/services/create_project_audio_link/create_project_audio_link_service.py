import re
from dataclasses import replace
from uuid import uuid4

from app.entities.project_audio_link.i_project_audio_links_repository import IProjectAudioLinksRepository
from app.entities.project_audio_link.project_audio_link_entity import ProjectAudioLinkEntity
from app.entities.project_audio_link.services.create_project_audio_link.dtos.create_project_audio_link_dto_in import (
    CreateProjectAudioLinkDtoIn,
)
from app.entities.project_audio_link.services.create_project_audio_link.dtos.create_project_audio_link_dto_out import (
    CreateProjectAudioLinkDtoOut,
)
from app.exceptions.invalid_input_error import InvalidInputError
from app.exceptions.resource_not_found_error import ResourceNotFoundError


class CreateProjectAudioLinkService:
    def __init__(self, repository: IProjectAudioLinksRepository):
        self.repository = repository

    def exec(self, dto: CreateProjectAudioLinkDtoIn) -> CreateProjectAudioLinkDtoOut:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,254}", dto.project_id):
            raise InvalidInputError()
        media = self.repository.owned_audio(dto.office_id, dto.owner_id, dto.output_id)
        if media is None:
            raise ResourceNotFoundError()
        entity = self.repository.find(dto.office_id, dto.project_id, media)
        create = entity is None
        if entity is not None and entity.created_by_user_id != dto.user_id:
            raise ResourceNotFoundError()
        entity = (
            ProjectAudioLinkEntity(str(uuid4()), dto.office_id, dto.project_id, media, dto.user_id)
            if create
            else replace(entity, status="active")
        )
        self.repository.save(entity, create)
        return CreateProjectAudioLinkDtoOut(entity)
