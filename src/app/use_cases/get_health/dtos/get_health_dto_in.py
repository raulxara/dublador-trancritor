from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class GetHealthDtoIn:
    check: Literal["liveness", "readiness"]
