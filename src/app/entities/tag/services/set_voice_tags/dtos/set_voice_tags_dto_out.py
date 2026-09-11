from dataclasses import dataclass

from app.entities.tag.tag_entity import TagEntity


@dataclass(frozen=True, slots=True)
class SetVoiceTagsDtoOut:
    data: tuple[TagEntity, ...]
