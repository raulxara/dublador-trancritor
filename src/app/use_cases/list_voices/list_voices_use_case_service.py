from dataclasses import asdict

from app.entities.voice.services.manage_voice.voice_service import VoiceService
from app.exceptions.authorization_error import AuthorizationError
from app.use_cases.list_voices.dtos.list_voices_dto_in import ListVoicesDtoIn
from app.use_cases.list_voices.dtos.list_voices_dto_out import ListVoicesDtoOut
from app.use_cases.manage_voice.manage_voice_use_case_service import ManageVoiceUseCaseService


class ListVoicesUseCaseService:
    def __init__(self, service: VoiceService, defaults=None):
        self.service = service
        self.defaults = defaults

    def exec(self, dto: ListVoicesDtoIn) -> ListVoicesDtoOut:
        if "voice.read" not in dto.actor.permissions:
            raise AuthorizationError()
        if self.defaults is not None:
            self.defaults.ensure(dto.actor.office_id)
        return ListVoicesDtoOut(
            [
                asdict(ManageVoiceUseCaseService.output(row))
                for row in self.service.list(dto.actor.office_id, dto.limit, dto.offset, dto.q, dto.language_id, dto.gender_id)
            ]
        )
