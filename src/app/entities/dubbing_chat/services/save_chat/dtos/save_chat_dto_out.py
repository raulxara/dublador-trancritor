from dataclasses import dataclass

from app.entities.dubbing_chat.dubbing_chat_entity import DubbingChatEntity


@dataclass(frozen=True, slots=True)
class SaveChatDtoOut:
    data: DubbingChatEntity
