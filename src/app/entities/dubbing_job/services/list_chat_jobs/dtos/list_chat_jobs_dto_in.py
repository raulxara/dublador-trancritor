from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListChatJobsDtoIn:
    office_id: str
    owner_id: str
    chat_id: str
    limit: int
    offset: int
