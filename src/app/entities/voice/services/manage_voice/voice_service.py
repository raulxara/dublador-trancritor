from uuid import uuid4

from app.entities.voice.i_voices_repository import IVoicesRepository
from app.entities.voice.voice_entity import VoiceEntity
from app.exceptions.resource_not_found_error import ResourceNotFoundError


class VoiceService:
    def __init__(self, repository: IVoicesRepository):
        self.repository = repository

    def create(self, office_id, name, language_id, gender_id, description):
        voice = VoiceEntity(str(uuid4()), office_id, name, language_id, gender_id, description=description)
        self.repository.save(voice, create=True)
        return voice

    def update(self, office_id, unique_id, name, language_id, gender_id, description, status):
        old = self.repository.find_by_unique_id(office_id, unique_id)
        if old is None:
            raise ResourceNotFoundError()
        voice = VoiceEntity(
            unique_id, office_id, name, language_id, gender_id, old.current_sample_id, status, description
        )
        self.repository.save(voice, create=False)
        return voice

    def list(self, office_id, limit, offset):
        return self.repository.list(office_id, limit, offset)
