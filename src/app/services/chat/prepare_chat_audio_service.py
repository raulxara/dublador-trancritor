from uuid import uuid4

from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.services.voice.validate_wav_service import ValidateWavService


class PrepareChatAudioService:
    def __init__(self, storage: IPrivateMediaStorage):
        self.storage = storage

    def exec(self, office_id: str, content: bytes) -> dict:
        metadata = ValidateWavService().exec(content)
        metadata["file"] = str(uuid4())
        metadata["key"] = self.storage.save(office_id, metadata["file"], content)
        return metadata
