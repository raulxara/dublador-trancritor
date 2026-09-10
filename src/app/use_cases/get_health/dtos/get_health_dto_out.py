from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GetHealthDtoOut:
    available: bool
    check: str
    service: str = "siplug-dubber"
