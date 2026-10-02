from dataclasses import dataclass, field

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class RegisterUserDtoIn:
    actor: AuthorizedActor
    first_name: str
    last_name: str
    email: str
    phone: str
    username: str
    password: str = field(repr=False)
    document_type: str | None = None
    document_value: str | None = None
    office_id: str | None = None
