from app.exceptions.authorization_error import AuthorizationError
from app.services.voice.voice_sample_service import VoiceSampleService
from app.use_cases.register_voice_sample.dtos.register_voice_sample_dto_in import RegisterVoiceSampleDtoIn
from app.use_cases.register_voice_sample.dtos.register_voice_sample_dto_out import RegisterVoiceSampleDtoOut


class RegisterVoiceSampleUseCaseService:
    def __init__(self, service: VoiceSampleService):
        self.service = service

    def exec(self, dto: RegisterVoiceSampleDtoIn) -> RegisterVoiceSampleDtoOut:
        if "voice.update" not in dto.actor.permissions:
            raise AuthorizationError()
        return RegisterVoiceSampleDtoOut(
            **self.service.register(dto.actor.office_id, dto.voice_id, dto.actor.user_id, dto.content)
        )
