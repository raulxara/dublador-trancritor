from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class ListVoicesDtoIn:
    actor: AuthorizedActor
    limit: int = 20
    offset: int = 0
