from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequireOwnedChatDtoIn:
    office_id: str
    owner_id: str
    chat_id: str
    allow_inactive: bool = False
