import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.interfaces.i_registration_repository import IRegistrationRepository
from app.use_cases.register_user.dtos.register_user_dto_in import RegisterUserDtoIn
from app.use_cases.register_user.dtos.register_user_dto_out import RegisterUserDtoOut


class RegisterUserService:
    def __init__(self, repository: IRegistrationRepository):
        self.repository = repository

    def exec(self, dto: RegisterUserDtoIn) -> RegisterUserDtoOut:
        token = secrets.token_hex(32)
        salt = secrets.token_hex(16)
        password = hashlib.scrypt(dto.password.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        values = {key: str(uuid4()) for key in ["profile", "user", "customer", "position", "user_position"]}
        values.update(
            office=dto.actor.office_id,
            first_name=dto.first_name,
            last_name=dto.last_name,
            email=dto.email.lower(),
            phone=dto.phone,
            username=dto.username,
            document_type=dto.document_type,
            document_value=dto.document_value,
            password=f"scrypt$16384$8$1${salt}${password}",
            token=hashlib.sha256(token.encode()).hexdigest(),
            expires=expires.replace(tzinfo=None),
        )
        result = self.repository.create_customer(values)
        return RegisterUserDtoOut(
            result.office_id,
            result.profile_id,
            result.user_id,
            result.customer_id,
            result.user_position_id,
            result.position_id,
            expires,
            token,
        )
