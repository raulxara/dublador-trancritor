from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GetVoiceDtoOut:
    data: dict
