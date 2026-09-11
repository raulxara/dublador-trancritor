from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class DownloadDubbingOutputDtoIn:
    actor: AuthorizedActor
    job_id: str
    output_id: str
