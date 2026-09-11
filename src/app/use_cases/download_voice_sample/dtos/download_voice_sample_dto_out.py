from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DownloadVoiceSampleDtoOut:
    data: str
