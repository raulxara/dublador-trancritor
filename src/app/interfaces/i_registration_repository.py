from typing import Protocol

from app.entities.user.registered_user_entity import RegisteredUserEntity


class IRegistrationRepository(Protocol):
    def create_customer(self, values: dict) -> RegisteredUserEntity: ...
