from app.entities.tag.i_tags_repository import ITagsRepository
from app.entities.tag.services.set_voice_tags.dtos.set_voice_tags_dto_in import SetVoiceTagsDtoIn
from app.entities.tag.services.set_voice_tags.dtos.set_voice_tags_dto_out import SetVoiceTagsDtoOut
from app.exceptions.invalid_input_error import InvalidInputError
from app.exceptions.resource_not_found_error import ResourceNotFoundError


class SetVoiceTagsService:
    def __init__(self, repository: ITagsRepository):
        self.repository = repository

    def exec(self, dto: SetVoiceTagsDtoIn) -> SetVoiceTagsDtoOut:
        if len(dto.tag_ids) > 50 or len(set(dto.tag_ids)) != len(dto.tag_ids):
            raise InvalidInputError()
        if not self.repository.voice_exists(dto.office_id, dto.voice_id):
            raise ResourceNotFoundError()
        for identifier in sorted(dto.tag_ids):
            tag = self.repository.find(dto.office_id, identifier)
            if tag is None or tag.status != "active":
                raise ResourceNotFoundError()
        self.repository.replace_voice(dto.office_id, dto.voice_id, dto.tag_ids)
        return SetVoiceTagsDtoOut(tuple(self.repository.list_voice(dto.office_id, dto.voice_id)))
