import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.interfaces.i_bootstrap_repository import IBootstrapRepository
from app.use_cases.bootstrap_access.dtos.bootstrap_access_dto_in import BootstrapAccessDtoIn
from app.use_cases.bootstrap_access.dtos.bootstrap_access_dto_out import BootstrapAccessDtoOut


class BootstrapAccessService:
    def __init__(self, repository: IBootstrapRepository):
        self.repository = repository

    def exec(self, dto: BootstrapAccessDtoIn) -> BootstrapAccessDtoOut:
        fields = [dto.office_name, dto.office_slug, dto.first_name, dto.last_name, dto.email, dto.username]
        if any(not v.strip() or len(v) > 255 for v in fields) or not 12 <= len(dto.password) <= 1024:
            raise ValueError("Informe campos não vazios, até 255 caracteres, e senha de 12 a 1024 caracteres.")
        token = secrets.token_hex(32)
        salt = secrets.token_hex(16)
        password = hashlib.scrypt(dto.password.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        values = {key: str(uuid4()) for key in ["office", "profile", "user", "customer", "position", "user_position"]}
        values.update(
            office_name=dto.office_name.strip(),
            office_slug=dto.office_slug.strip(),
            first_name=dto.first_name.strip(),
            last_name=dto.last_name.strip(),
            email=dto.email.strip().lower(),
            username=dto.username.strip(),
            password=f"scrypt$16384$8$1${salt}${password}",
            token=hashlib.sha256(token.encode()).hexdigest(),
            expires=expires.replace(tzinfo=None),
        )
        self.repository.create_initial_access(values)
        return BootstrapAccessDtoOut(values["office"], values["user"], token, expires)
