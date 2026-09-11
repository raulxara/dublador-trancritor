from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UpdateDubbingChatDtoOut:
    data: dict
