import re
from dataclasses import replace

from app.entities.tag.i_tags_repository import ITagsRepository
from app.entities.tag.services.update_tag.dtos.update_tag_dto_in import UpdateTagDtoIn
from app.entities.tag.services.update_tag.dtos.update_tag_dto_out import UpdateTagDtoOut
from app.exceptions.invalid_input_error import InvalidInputError
from app.exceptions.resource_not_found_error import ResourceNotFoundError


class UpdateTagService:
    def __init__(self, repository: ITagsRepository):
        self.repository = repository

    def exec(self, dto: UpdateTagDtoIn) -> UpdateTagDtoOut:
        entity = self.repository.find(dto.office_id, dto.tag_id)
        if entity is None:
            raise ResourceNotFoundError()
        name = dto.name.strip()
        if (
            not name
            or len(name) > 255
            or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", dto.slug)
            or len(dto.slug) > 255
            or dto.status not in ("active", "inactive")
        ):
            raise InvalidInputError()
        entity = replace(entity, name=name, slug=dto.slug, status=dto.status)
        self.repository.save(entity, False)
        return UpdateTagDtoOut(entity)
