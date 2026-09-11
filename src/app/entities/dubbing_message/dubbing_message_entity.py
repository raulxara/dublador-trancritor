from dataclasses import dataclass

from app.exceptions.invalid_input_error import InvalidInputError


@dataclass(frozen=True, slots=True)
class DubbingMessageEntity:
    unique_id: str
    office_id: str
    chat_id: str
    user_customer_id: str | None
    role: str
    message_type: str
    content: str | None
    status: str = "active"

    def __post_init__(self):
        if (
            not self.unique_id
            or not self.office_id
            or not self.chat_id
            or self.role not in ("user", "assistant", "system")
            or self.message_type not in ("text", "audio", "video", "result", "error")
            or self.status not in ("active", "inactive")
        ):
            raise InvalidInputError()
