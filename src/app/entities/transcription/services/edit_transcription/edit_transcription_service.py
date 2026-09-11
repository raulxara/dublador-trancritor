from dataclasses import replace
from uuid import uuid4

from app.entities.transcription.i_transcriptions_repository import ITranscriptionsRepository
from app.entities.transcription.services.edit_transcription.dtos.edit_transcription_dto_in import EditTranscriptionDtoIn
from app.entities.transcription.services.edit_transcription.dtos.edit_transcription_dto_out import (
    EditTranscriptionDtoOut,
)
from app.exceptions.conflict_error import ConflictError
from app.exceptions.invalid_input_error import InvalidInputError
from app.exceptions.resource_not_found_error import ResourceNotFoundError


class EditTranscriptionService:
    def __init__(self, repository: ITranscriptionsRepository):
        self.repository = repository

    def exec(self, dto: EditTranscriptionDtoIn) -> EditTranscriptionDtoOut:
        current = self.repository.latest(dto.office_id, dto.job_id)
        if current is None or current.status != "active":
            raise ResourceNotFoundError()
        if current.unique_id != dto.base_transcription_id:
            raise ConflictError()
        content = dto.text.strip()
        if not content or len(content) > 10000:
            raise InvalidInputError()
        entity = replace(
            current,
            unique_id=str(uuid4()),
            text=content,
            version=current.version + 1,
            origin="edited",
            edited_by_user_id=dto.user_id,
            previous_transcription_id=current.unique_id,
        )
        self.repository.create(entity)
        return EditTranscriptionDtoOut(entity)
