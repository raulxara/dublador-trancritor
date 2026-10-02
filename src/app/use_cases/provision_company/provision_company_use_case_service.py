from app.exceptions.authorization_error import AuthorizationError
from app.services.actor.provision_company_service import ProvisionCompanyService
from app.use_cases.provision_company.dtos.provision_company_dto_in import ProvisionCompanyDtoIn

class ProvisionCompanyUseCaseService:
    def __init__(self, service: ProvisionCompanyService, allowed_office: str):
        self.service = service
        self.allowed_office = allowed_office

    def exec(self, dto: ProvisionCompanyDtoIn):
        if not self.allowed_office or dto.registration.actor.office_id != self.allowed_office or "office.provision" not in dto.registration.actor.permissions:
            raise AuthorizationError()
        return self.service.exec(dto)
