from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RegisterUserDtoOut:
    office_id: str
    profile_id: str
    user_id: str
    user_customer_id: str
    user_position_id: str
    position_id: str
    expires_at: datetime
    token: str = field(repr=False)
