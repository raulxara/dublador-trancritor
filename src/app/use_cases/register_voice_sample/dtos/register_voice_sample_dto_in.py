from dataclasses import dataclass, field

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class RegisterVoiceSampleDtoIn:
    actor: AuthorizedActor
    voice_id: str
    content: bytes = field(repr=False)
