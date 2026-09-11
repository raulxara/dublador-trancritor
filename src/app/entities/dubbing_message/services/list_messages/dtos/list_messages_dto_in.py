from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListMessagesDtoIn:
    office_id: str
    chat_id: str
    limit: int
    offset: int
