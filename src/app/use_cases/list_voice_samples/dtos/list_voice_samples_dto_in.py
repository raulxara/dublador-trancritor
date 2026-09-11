from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class ListVoiceSamplesDtoIn:
    actor: AuthorizedActor
    voice_id: str
    limit: int = 20
    offset: int = 0
