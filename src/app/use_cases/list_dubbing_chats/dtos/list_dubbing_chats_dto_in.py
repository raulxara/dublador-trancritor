from dataclasses import dataclass

from app.exceptions.invalid_input_error import InvalidInputError
from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class ListDubbingChatsDtoIn:
    actor: AuthorizedActor
    limit: int = 20
    offset: int = 0

    def __post_init__(self):
        if not 1 <= self.limit <= 100 or self.offset < 0:
            raise InvalidInputError()
