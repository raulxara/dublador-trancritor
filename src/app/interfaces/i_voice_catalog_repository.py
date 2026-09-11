from typing import Protocol


class IVoiceCatalogRepository(Protocol):
    def list(self, catalog: str) -> list[dict]: ...
