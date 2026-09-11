import re

from app.entities.project_audio_link.i_project_audio_links_repository import IProjectAudioLinksRepository
from app.entities.project_audio_link.services.list_project_audio_links.dtos.list_project_audio_links_dto_in import (
    ListProjectAudioLinksDtoIn,
)
from app.entities.project_audio_link.services.list_project_audio_links.dtos.list_project_audio_links_dto_out import (
    ListProjectAudioLinksDtoOut,
)
from app.exceptions.invalid_input_error import InvalidInputError


class ListProjectAudioLinksService:
    def __init__(self, repository: IProjectAudioLinksRepository):
        self.repository = repository

    def exec(self, dto: ListProjectAudioLinksDtoIn) -> ListProjectAudioLinksDtoOut:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,254}", dto.project_id):
            raise InvalidInputError()
        if not 1 <= dto.limit <= 100 or dto.offset < 0:
            raise InvalidInputError()
        return ListProjectAudioLinksDtoOut(
            tuple(self.repository.list(dto.office_id, dto.user_id, dto.project_id, dto.limit, dto.offset))
        )
