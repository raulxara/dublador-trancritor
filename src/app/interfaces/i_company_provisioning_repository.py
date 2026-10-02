from typing import Protocol
from app.entities.user.registered_user_entity import RegisteredUserEntity

class ICompanyProvisioningRepository(Protocol):
    def create(self, source: str, fingerprint: str, values: dict) -> RegisteredUserEntity: ...
