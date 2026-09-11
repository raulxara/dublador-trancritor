from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class CreateProjectAudioLinkDtoIn:
    actor: AuthorizedActor
    project_id: str
    output_id: str
