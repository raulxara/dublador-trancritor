from dataclasses import dataclass

from app.exceptions.invalid_input_error import InvalidInputError


@dataclass(frozen=True, slots=True)
class DubbingChatEntity:
    unique_id: str
    office_id: str
    user_customer_id: str
    title: str | None
    selected_voice_id: str | None
    status: str = "active"

    def __post_init__(self):
        if (
            not self.unique_id
            or not self.office_id
            or not self.user_customer_id
            or self.status not in ("active", "inactive")
        ):
            raise InvalidInputError()
