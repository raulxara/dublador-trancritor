from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TagEntity:
    unique_id: str
    office_id: str
    name: str
    slug: str
    status: str = "active"
