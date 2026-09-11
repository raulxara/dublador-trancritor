from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GetDubbingChatDtoOut:
    data: dict
