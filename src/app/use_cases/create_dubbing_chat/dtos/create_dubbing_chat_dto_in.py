from dataclasses import dataclass

from app.exceptions.invalid_input_error import InvalidInputError
from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class CreateDubbingChatDtoIn:
    actor: AuthorizedActor
    title: str | None = None
    selected_voice_id: str | None = None

    def __post_init__(self):
        if self.title is not None and (not self.title.strip() or len(self.title) > 255):
            raise InvalidInputError()
