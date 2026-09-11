import re
from uuid import uuid4

from app.entities.tag.i_tags_repository import ITagsRepository
from app.entities.tag.services.create_tag.dtos.create_tag_dto_in import CreateTagDtoIn
from app.entities.tag.services.create_tag.dtos.create_tag_dto_out import CreateTagDtoOut
from app.entities.tag.tag_entity import TagEntity
from app.exceptions.invalid_input_error import InvalidInputError


class CreateTagService:
    def __init__(self, repository: ITagsRepository):
        self.repository = repository

    def exec(self, dto: CreateTagDtoIn) -> CreateTagDtoOut:
        name = dto.name.strip()
        if (
            not name
            or len(name) > 255
            or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", dto.slug)
            or len(dto.slug) > 255
        ):
            raise InvalidInputError()
        entity = TagEntity(str(uuid4()), dto.office_id, name, dto.slug)
        self.repository.save(entity, True)
        return CreateTagDtoOut(entity)
