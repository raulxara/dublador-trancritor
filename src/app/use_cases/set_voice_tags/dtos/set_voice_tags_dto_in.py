from dataclasses import dataclass

from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class SetVoiceTagsDtoIn:
    actor: AuthorizedActor
    voice_id: str
    tag_ids: tuple[str, ...]
