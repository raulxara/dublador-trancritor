from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UpdateTagDtoIn:
    office_id: str
    user_id: str
    owner_id: str
    tag_id: str
    name: str
    slug: str
    status: str
