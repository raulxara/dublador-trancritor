from dataclasses import dataclass, field

from app.exceptions.invalid_input_error import InvalidInputError
from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class UploadChatAudioDtoIn:
    actor: AuthorizedActor
    chat_id: str
    content: bytes = field(repr=False)

    def __post_init__(self):
        if not self.chat_id or len(self.chat_id) > 255:
            raise InvalidInputError()

    content_type: str = "audio/wav"
