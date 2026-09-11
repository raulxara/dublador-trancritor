from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CreateTagDtoIn:
    office_id: str
    user_id: str
    owner_id: str
    name: str
    slug: str
