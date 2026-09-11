from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CleanupOrphanMediaDtoIn:
    minimum_age_seconds: int = 86400
    limit: int = 500
