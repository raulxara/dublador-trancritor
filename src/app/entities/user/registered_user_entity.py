from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegisteredUserEntity:
    office_id: str
    profile_id: str
    user_id: str
    customer_id: str
    user_position_id: str
    position_id: str
