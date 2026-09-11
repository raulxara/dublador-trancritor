from app.services.actor.bootstrap_access_service import BootstrapAccessService
from app.use_cases.bootstrap_access.dtos.bootstrap_access_dto_in import BootstrapAccessDtoIn
from app.use_cases.bootstrap_access.dtos.bootstrap_access_dto_out import BootstrapAccessDtoOut


class BootstrapAccessUseCaseService:
    def __init__(self, service: BootstrapAccessService):
        self.service = service

    def exec(self, dto: BootstrapAccessDtoIn) -> BootstrapAccessDtoOut:
        return self.service.exec(dto)
