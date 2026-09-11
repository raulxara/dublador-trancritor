from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class EditTranscriptionDtoIn:
    actor: AuthorizedActor
    job_id: str
    base_transcription_id: str
    text: str
