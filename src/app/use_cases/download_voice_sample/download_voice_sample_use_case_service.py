from app.exceptions.authorization_error import AuthorizationError
from app.services.voice.voice_sample_service import VoiceSampleService
from app.use_cases.download_voice_sample.dtos.download_voice_sample_dto_in import DownloadVoiceSampleDtoIn
from app.use_cases.download_voice_sample.dtos.download_voice_sample_dto_out import DownloadVoiceSampleDtoOut


class DownloadVoiceSampleUseCaseService:
    def __init__(self, service: VoiceSampleService):
        self.service = service

    def exec(self, dto: DownloadVoiceSampleDtoIn) -> DownloadVoiceSampleDtoOut:
        if "voice.read" not in dto.actor.permissions:
            raise AuthorizationError()
        return DownloadVoiceSampleDtoOut(self.service.download(dto.actor.office_id, dto.voice_id, dto.sample_id))
