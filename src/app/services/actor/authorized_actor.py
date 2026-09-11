from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuthorizedActor:
    office_id: str
    user_id: str
    user_customer_id: str
    profile_id: str | None
    permissions: frozenset[str]
