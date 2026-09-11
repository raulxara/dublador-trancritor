from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ListProjectAudioLinksDtoOut:
    data: list[dict]
