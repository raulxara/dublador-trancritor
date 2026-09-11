from typing import Protocol


class IBootstrapRepository(Protocol):
    def create_initial_access(self, values: dict) -> None: ...
