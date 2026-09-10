import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from app.exceptions.resource_not_found_error import ResourceNotFoundError
from app.interfaces.i_authorization_repository import IAuthorizationRepository


class AccessTokenService:
    def __init__(self, repository: IAuthorizationRepository):
        self.repository = repository

    def issue(self, office_id: str, user_id: str) -> tuple[str, datetime]:
        token = secrets.token_hex(32)
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        if not self.repository.replace_token(
            office_id, user_id, hashlib.sha256(token.encode()).hexdigest(), expires.replace(tzinfo=None)
        ):
            raise ResourceNotFoundError()
        return token, expires

    def revoke(self, office_id: str, user_customer_id: str) -> None:
        self.repository.revoke_token(office_id, user_customer_id)
