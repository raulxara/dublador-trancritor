from types import TracebackType
from typing import Protocol, Self


class IUnitOfWork(Protocol):
    """Repositories share a transaction; exit without commit rolls back."""

    def __enter__(self) -> Self: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
