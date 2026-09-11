from app.exceptions.authorization_error import AuthorizationError
from app.services.voice.voice_sample_service import VoiceSampleService
from app.use_cases.list_voice_samples.dtos.list_voice_samples_dto_in import ListVoiceSamplesDtoIn
from app.use_cases.list_voice_samples.dtos.list_voice_samples_dto_out import ListVoiceSamplesDtoOut


class ListVoiceSamplesUseCaseService:
    def __init__(self, service: VoiceSampleService):
        self.service = service

    def exec(self, dto: ListVoiceSamplesDtoIn) -> ListVoiceSamplesDtoOut:
        if "voice.read" not in dto.actor.permissions:
            raise AuthorizationError()
        return ListVoiceSamplesDtoOut(self.service.list(dto.actor.office_id, dto.voice_id, dto.limit, dto.offset))
