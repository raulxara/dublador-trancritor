from dataclasses import dataclass
from app.use_cases.register_user.dtos.register_user_dto_in import RegisterUserDtoIn

@dataclass(frozen=True, slots=True)
class ProvisionCompanyDtoIn:
    registration: RegisterUserDtoIn
    key: str
    name: str
    slug: str
