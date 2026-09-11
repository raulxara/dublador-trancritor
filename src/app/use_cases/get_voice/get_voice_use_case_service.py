from dataclasses import asdict

from app.entities.voice.services.find_voice_by_unique_id.dtos.find_voice_by_unique_id_dto_in import (
    FindVoiceByUniqueIdDtoIn,
)
from app.entities.voice.services.find_voice_by_unique_id.find_voice_by_unique_id_service import (
    FindVoiceByUniqueIdService,
)
from app.exceptions.authorization_error import AuthorizationError
from app.use_cases.get_voice.dtos.get_voice_dto_in import GetVoiceDtoIn
from app.use_cases.get_voice.dtos.get_voice_dto_out import GetVoiceDtoOut


class GetVoiceUseCaseService:
    def __init__(self, service: FindVoiceByUniqueIdService):
        self.service = service

    def exec(self, dto: GetVoiceDtoIn) -> GetVoiceDtoOut:
        if "voice.read" not in dto.actor.permissions:
            raise AuthorizationError()
        return GetVoiceDtoOut(asdict(self.service.exec(FindVoiceByUniqueIdDtoIn(dto.actor.office_id, dto.voice_id))))
