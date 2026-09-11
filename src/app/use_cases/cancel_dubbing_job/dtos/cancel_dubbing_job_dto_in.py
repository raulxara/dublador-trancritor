from dataclasses import dataclass

from app.exceptions.invalid_input_error import InvalidInputError
from app.services.actor.authorized_actor import AuthorizedActor


@dataclass(frozen=True, slots=True)
class CancelDubbingJobDtoIn:
    actor: AuthorizedActor
    job_id: str

    def __post_init__(self):
        if not self.job_id or len(self.job_id) > 255:
            raise InvalidInputError()
