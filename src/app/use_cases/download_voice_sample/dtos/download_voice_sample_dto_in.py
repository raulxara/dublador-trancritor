from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class DownloadVoiceSampleDtoIn:
    actor: AuthorizedActor
    voice_id: str
    sample_id: str
