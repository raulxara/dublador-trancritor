from uuid import uuid4

from app.entities.voice.i_voices_repository import IVoicesRepository
from app.exceptions.resource_not_found_error import ResourceNotFoundError
from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.interfaces.i_sample_repository import ISampleRepository
from app.services.voice.validate_wav_service import ValidateWavService


class VoiceSampleService:
    def __init__(self, voices: IVoicesRepository, samples: ISampleRepository, storage: IPrivateMediaStorage):
        self.voices, self.samples, self.storage = voices, samples, storage

    def require_voice(self, office_id, voice_id):
        voice = self.voices.find_by_unique_id(office_id, voice_id)
        if voice is None or voice.status != "active":
            raise ResourceNotFoundError()

    def register(self, office_id, voice_id, user_id, content):
        self.require_voice(office_id, voice_id)
        metadata = ValidateWavService().exec(content)
        metadata.update(file=str(uuid4()), sample=str(uuid4()))
        metadata["key"] = self.storage.save(office_id, metadata["file"], content)
        # Preserve a private orphan on an ambiguous DB commit rather than delete a possibly committed file.
        return self.samples.register(office_id, voice_id, user_id, metadata)

    def list(self, office_id, voice_id, limit, offset):
        self.require_voice(office_id, voice_id)
        return self.samples.list(office_id, voice_id, limit, offset)

    def download(self, office_id, voice_id, sample_id):
        key = self.samples.find_file(office_id, voice_id, sample_id)
        if key is None:
            raise ResourceNotFoundError()
        return self.storage.path(key)
