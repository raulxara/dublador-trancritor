from dataclasses import dataclass

from app.entities.dubbing_message.dubbing_message_entity import DubbingMessageEntity


@dataclass(frozen=True, slots=True)
class CreateMessageDtoIn:
    message: DubbingMessageEntity
