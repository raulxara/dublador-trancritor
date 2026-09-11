from typing import Protocol

from app.entities.voice.voice_entity import VoiceEntity


class IVoicesRepository(Protocol):
    def find_by_unique_id(self, office_id: str, unique_id: str) -> VoiceEntity | None:
        """Filter by office_id and _id in the query, not after serialization."""
        ...

    def save(self, voice: VoiceEntity, create: bool) -> None: ...
    def list(self, office_id: str, limit: int, offset: int) -> list[VoiceEntity]: ...
