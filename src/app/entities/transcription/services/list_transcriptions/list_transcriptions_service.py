from app.entities.transcription.i_transcriptions_repository import ITranscriptionsRepository
from app.entities.transcription.services.list_transcriptions.dtos.list_transcriptions_dto_in import (
    ListTranscriptionsDtoIn,
)
from app.entities.transcription.services.list_transcriptions.dtos.list_transcriptions_dto_out import (
    ListTranscriptionsDtoOut,
)
from app.exceptions.invalid_input_error import InvalidInputError


class ListTranscriptionsService:
    def __init__(self, repository: ITranscriptionsRepository):
        self.repository = repository

    def exec(self, dto: ListTranscriptionsDtoIn) -> ListTranscriptionsDtoOut:
        if not 1 <= dto.limit <= 100 or dto.offset < 0:
            raise InvalidInputError()
        return ListTranscriptionsDtoOut(tuple(self.repository.list(dto.office_id, dto.job_id, dto.limit, dto.offset)))
