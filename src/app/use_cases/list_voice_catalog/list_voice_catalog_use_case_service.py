from app.exceptions.authorization_error import AuthorizationError
from app.services.voice.voice_catalog_service import VoiceCatalogService
from app.use_cases.list_voice_catalog.dtos.list_voice_catalog_dto_in import ListVoiceCatalogDtoIn
from app.use_cases.list_voice_catalog.dtos.list_voice_catalog_dto_out import ListVoiceCatalogDtoOut


class ListVoiceCatalogUseCaseService:
    def __init__(self, service: VoiceCatalogService):
        self.service = service

    def exec(self, dto: ListVoiceCatalogDtoIn) -> ListVoiceCatalogDtoOut:
        if "catalog.read" not in dto.actor.permissions:
            raise AuthorizationError()
        return ListVoiceCatalogDtoOut(self.service.list(dto.catalog))
