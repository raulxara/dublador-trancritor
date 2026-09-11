from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListVoiceCatalogDtoOut:
    data: list[dict]
