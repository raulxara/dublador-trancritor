from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class ListTagsDtoIn:
    actor: AuthorizedActor
    limit: int
    offset: int
