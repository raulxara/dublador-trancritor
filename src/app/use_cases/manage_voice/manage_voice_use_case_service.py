from app.entities.voice.services.manage_voice.voice_service import VoiceService
from app.exceptions.authorization_error import AuthorizationError
from app.exceptions.invalid_input_error import InvalidInputError
from app.use_cases.manage_voice.dtos.manage_voice_dto_in import ManageVoiceDtoIn
from app.use_cases.manage_voice.dtos.manage_voice_dto_out import ManageVoiceDtoOut


class ManageVoiceUseCaseService:
    def __init__(self, service: VoiceService):
        self.service = service

    @staticmethod
    def output(voice):
        return ManageVoiceDtoOut(
            voice.unique_id,
            voice.name,
            voice.language_id,
            voice.gender_id,
            voice.current_sample_id,
            voice.status,
            voice.description,
        )

    def exec(self, dto: ManageVoiceDtoIn) -> ManageVoiceDtoOut:
        permission = "voice.update" if dto.unique_id else "voice.register"
        if permission not in dto.actor.permissions:
            raise AuthorizationError()
        if not dto.name.strip() or len(dto.name) > 255 or dto.status not in ("active", "inactive"):
            raise InvalidInputError()
        if dto.unique_id:
            voice = self.service.update(
                dto.actor.office_id,
                dto.unique_id,
                dto.name.strip(),
                dto.language_id,
                dto.gender_id,
                dto.description,
                dto.status,
            )
        else:
            voice = self.service.create(
                dto.actor.office_id, dto.name.strip(), dto.language_id, dto.gender_id, dto.description
            )
        return self.output(voice)
