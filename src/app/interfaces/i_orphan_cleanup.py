from typing import Protocol


class IOrphanCleanup(Protocol):
    def exec(self, minimum_age_seconds: int, limit: int) -> int: ...
