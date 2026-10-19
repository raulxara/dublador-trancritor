from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from app.services.voice.validate_wav_service import ValidateWavService


class DefaultVoiceService:
    """Install the operator-provided reference independently for each tenant."""

    def __init__(self, repository, storage, source):
        self.repository, self.storage, self.source = repository, storage, Path(source)

    def ensure(self, office_id):
        voice_id = str(uuid5(NAMESPACE_URL, "siplug:default:adam:v1:" + office_id))
        if self.repository.exists(office_id, voice_id) or not self.source.is_file():
            return
        content = self.source.read_bytes()
        metadata = ValidateWavService().exec(content)
        metadata.update(voice=voice_id, file=str(uuid5(NAMESPACE_URL, voice_id + ":file")),
                        sample=str(uuid5(NAMESPACE_URL, voice_id + ":sample")))
        with self.storage.publication():
            metadata["key"] = self.storage.save(office_id, metadata["file"], content)
            self.repository.install(office_id, metadata)
