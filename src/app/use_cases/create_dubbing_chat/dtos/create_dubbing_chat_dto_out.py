from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CreateDubbingChatDtoOut:
    data: dict
