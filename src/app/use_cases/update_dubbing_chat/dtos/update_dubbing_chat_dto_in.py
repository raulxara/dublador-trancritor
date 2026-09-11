from dataclasses import dataclass

from app.exceptions.invalid_input_error import InvalidInputError
from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class UpdateDubbingChatDtoIn:
    actor: AuthorizedActor
    chat_id: str
    title: str | None = None
    selected_voice_id: str | None = None
    status: str = "active"

    def __post_init__(self):
        if (self.title is not None and (not self.title.strip() or len(self.title) > 255)) or (
            not self.chat_id or len(self.chat_id) > 255
        ):
            raise InvalidInputError()
