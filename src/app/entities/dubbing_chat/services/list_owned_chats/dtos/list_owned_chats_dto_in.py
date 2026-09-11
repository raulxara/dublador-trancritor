from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListOwnedChatsDtoIn:
    office_id: str
    owner_id: str
    limit: int
    offset: int
