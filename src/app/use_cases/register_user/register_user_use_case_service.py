from app.exceptions.authorization_error import AuthorizationError
from app.services.actor.register_user_service import RegisterUserService
from app.use_cases.register_user.dtos.register_user_dto_in import RegisterUserDtoIn
from app.use_cases.register_user.dtos.register_user_dto_out import RegisterUserDtoOut


class RegisterUserUseCaseService:
    def __init__(self, service: RegisterUserService):
        self.service = service

    def exec(self, dto: RegisterUserDtoIn) -> RegisterUserDtoOut:
        if "user.register" not in dto.actor.permissions:
            raise AuthorizationError()
        if dto.office_id is not None and dto.office_id != dto.actor.office_id:
            raise AuthorizationError()
        return self.service.exec(dto)
