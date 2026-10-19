from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class ListVoicesDtoIn:
    actor: AuthorizedActor
    limit: int = 20
    offset: int = 0

    q: str = ""
    language_id: str | None = None
    gender_id: str | None = None
