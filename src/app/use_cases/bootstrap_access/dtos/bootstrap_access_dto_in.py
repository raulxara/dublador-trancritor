from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BootstrapAccessDtoIn:
    office_name: str
    office_slug: str
    first_name: str
    last_name: str
    email: str
    username: str
    password: str = field(repr=False)
