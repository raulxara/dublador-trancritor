from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class UpdateTagDtoIn:
    actor: AuthorizedActor
    tag_id: str
    name: str
    slug: str
    status: str
