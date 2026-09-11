from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class ListTranscriptionsDtoIn:
    actor: AuthorizedActor
    job_id: str
    limit: int
    offset: int
