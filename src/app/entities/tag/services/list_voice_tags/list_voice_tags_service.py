from app.entities.tag.i_tags_repository import ITagsRepository
from app.entities.tag.services.list_voice_tags.dtos.list_voice_tags_dto_in import ListVoiceTagsDtoIn
from app.entities.tag.services.list_voice_tags.dtos.list_voice_tags_dto_out import ListVoiceTagsDtoOut
from app.exceptions.resource_not_found_error import ResourceNotFoundError


class ListVoiceTagsService:
    def __init__(self, repository: ITagsRepository):
        self.repository = repository

    def exec(self, dto: ListVoiceTagsDtoIn) -> ListVoiceTagsDtoOut:
        if not self.repository.voice_exists(dto.office_id, dto.voice_id):
            raise ResourceNotFoundError()
        return ListVoiceTagsDtoOut(tuple(self.repository.list_voice(dto.office_id, dto.voice_id)))
