from dataclasses import dataclass

from app.entities.tag.tag_entity import TagEntity


@dataclass(frozen=True, slots=True)
class ListVoiceTagsDtoOut:
    data: tuple[TagEntity, ...]
