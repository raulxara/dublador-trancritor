from typing import Protocol


class IMediaInventoryRepository(Protocol):
    def referenced(self, key: str) -> bool: ...
