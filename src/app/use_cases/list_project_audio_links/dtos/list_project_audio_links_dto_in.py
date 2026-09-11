from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class ListProjectAudioLinksDtoIn:
    actor: AuthorizedActor
    project_id: str
    limit: int
    offset: int
