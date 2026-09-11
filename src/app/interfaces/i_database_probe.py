from typing import Protocol


class IDatabaseProbe(Protocol):
    def is_available(self) -> bool: ...
