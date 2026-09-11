from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class UnlinkProjectAudioDtoIn:
    actor: AuthorizedActor
    link_id: str
