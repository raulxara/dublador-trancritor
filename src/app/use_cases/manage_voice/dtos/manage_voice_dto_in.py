from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class ManageVoiceDtoIn:
    actor: AuthorizedActor
    name: str
    language_id: str
    gender_id: str | None = None
    description: str | None = None
    unique_id: str | None = None
    status: str = "active"
