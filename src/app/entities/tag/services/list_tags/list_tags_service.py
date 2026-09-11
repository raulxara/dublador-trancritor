from app.entities.tag.i_tags_repository import ITagsRepository
from app.entities.tag.services.list_tags.dtos.list_tags_dto_in import ListTagsDtoIn
from app.entities.tag.services.list_tags.dtos.list_tags_dto_out import ListTagsDtoOut
from app.exceptions.invalid_input_error import InvalidInputError


class ListTagsService:
    def __init__(self, repository: ITagsRepository):
        self.repository = repository

    def exec(self, dto: ListTagsDtoIn) -> ListTagsDtoOut:
        if not 1 <= dto.limit <= 100 or dto.offset < 0:
            raise InvalidInputError()
        return ListTagsDtoOut(tuple(self.repository.list(dto.office_id, dto.limit, dto.offset)))
