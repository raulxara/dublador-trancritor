from app.exceptions.invalid_input_error import InvalidInputError
from app.exceptions.resource_not_found_error import ResourceNotFoundError
from app.interfaces.i_chat_source_repository import IChatSourceRepository
from app.interfaces.i_private_media_storage import IPrivateMediaStorage


class ReadyChatSourceService:
    def __init__(self, repository: IChatSourceRepository, storage: IPrivateMediaStorage):
        self.repository, self.storage = repository, storage

    def sample(self, office_id: str, voice_id: str | None) -> dict:
        if voice_id is None:
            raise InvalidInputError()
        source = self.repository.sample(office_id, voice_id)
        if source is None:
            raise ResourceNotFoundError()
        self.storage.path(source["storage_key"])
        return source

    def input(self, office_id: str, message_id: str) -> dict:
        source = self.repository.input_file(office_id, message_id)
        if source is None:
            raise ResourceNotFoundError()
        self.storage.path(source["storage_key"])
        return source

    def language(self, language_id: str | None) -> None:
        if language_id is not None and not self.repository.language_active(language_id):
            raise InvalidInputError()
