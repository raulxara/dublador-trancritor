from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class BootstrapAccessDtoOut:
    office_id: str
    user_id: str
    token: str = field(repr=False)
    expires_at: datetime
