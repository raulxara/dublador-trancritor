from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequireSourceMessageDtoIn:
    office_id: str
    chat_id: str
    owner_id: str
    message_id: str
