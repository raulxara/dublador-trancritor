import hashlib
import os
from pathlib import Path
from uuid import UUID, uuid4

from app.exceptions.resource_not_found_error import ResourceNotFoundError
from app.services.media.media_publication_guard import MediaPublicationGuard


class PrivateMediaStorage:
    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def path(self, key: str) -> str:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ResourceNotFoundError()
        return str(path)

    def save(self, office_id: str, file_id: str, content: bytes) -> str:
        key = hashlib.sha256(office_id.encode()).hexdigest() + "/" + str(UUID(file_id)) + ".wav"
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = target.with_suffix("." + str(uuid4()) + ".tmp")
        try:
            with temporary.open("xb") as stream:
                os.chmod(temporary, 0o600)
                stream.write(content)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return key

    def publication(self):
        return MediaPublicationGuard(self.root).hold()
