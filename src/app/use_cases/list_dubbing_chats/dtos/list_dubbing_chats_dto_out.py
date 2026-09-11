from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListDubbingChatsDtoOut:
    data: list[dict]
