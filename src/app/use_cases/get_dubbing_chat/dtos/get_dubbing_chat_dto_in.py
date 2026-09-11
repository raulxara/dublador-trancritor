from dataclasses import dataclass

from app.exceptions.invalid_input_error import InvalidInputError
from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class GetDubbingChatDtoIn:
    actor: AuthorizedActor
    chat_id: str

    def __post_init__(self):
        if not self.chat_id or len(self.chat_id) > 255:
            raise InvalidInputError()
