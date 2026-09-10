from app.entities.voice.i_voices_repository import IVoicesRepository
from app.entities.voice.services.find_voice_by_unique_id.dtos.find_voice_by_unique_id_dto_in import (
    FindVoiceByUniqueIdDtoIn,
)
from app.entities.voice.services.find_voice_by_unique_id.dtos.find_voice_by_unique_id_dto_out import (
    FindVoiceByUniqueIdDtoOut,
)
from app.exceptions.resource_not_found_error import ResourceNotFoundError


class FindVoiceByUniqueIdService:
    def __init__(self, repository: IVoicesRepository) -> None:
        self.repository = repository

    def exec(self, dto_in: FindVoiceByUniqueIdDtoIn) -> FindVoiceByUniqueIdDtoOut:
        voice = self.repository.find_by_unique_id(dto_in.office_id, dto_in.unique_id)
        if voice is None or voice.office_id != dto_in.office_id or voice.unique_id != dto_in.unique_id:
            raise ResourceNotFoundError()
        return FindVoiceByUniqueIdDtoOut(
            unique_id=voice.unique_id,
            office_id=voice.office_id,
            name=voice.name,
            language_id=voice.language_id,
            gender_id=voice.gender_id,
            current_sample_id=voice.current_sample_id,
            status=voice.status,
        )
