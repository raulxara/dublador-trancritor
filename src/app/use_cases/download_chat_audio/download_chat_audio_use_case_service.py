from typing import Callable

from app.entities.dubbing_chat.services.require_owned_chat.dtos.require_owned_chat_dto_in import RequireOwnedChatDtoIn
from app.entities.dubbing_chat.services.require_owned_chat.require_owned_chat_service import RequireOwnedChatService
from app.entities.dubbing_message.services.require_source_message.dtos.require_source_message_dto_in import (
    RequireSourceMessageDtoIn,
)
from app.entities.dubbing_message.services.require_source_message.require_source_message_service import (
    RequireSourceMessageService,
)
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.services.chat.ready_chat_source_service import ReadyChatSourceService
from app.use_cases.download_chat_audio.dtos.download_chat_audio_dto_in import DownloadChatAudioDtoIn
from app.use_cases.download_chat_audio.dtos.download_chat_audio_dto_out import DownloadChatAudioDtoOut


class DownloadChatAudioUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork], storage: IPrivateMediaStorage):
        self.uow_factory, self.storage = uow_factory, storage

    def exec(self, dto: DownloadChatAudioDtoIn) -> DownloadChatAudioDtoOut:
        if "dubbing.download" not in dto.actor.permissions:
            raise AuthorizationError()

        with self.uow_factory() as uow:
            RequireOwnedChatService(uow.chats).exec(
                RequireOwnedChatDtoIn(dto.actor.office_id, dto.actor.user_customer_id, dto.chat_id)
            ).data
            RequireSourceMessageService(uow.messages).exec(
                RequireSourceMessageDtoIn(dto.actor.office_id, dto.chat_id, dto.actor.user_customer_id, dto.message_id)
            )
            source = ReadyChatSourceService(uow.sources, self.storage).input(dto.actor.office_id, dto.message_id)
            return DownloadChatAudioDtoOut(self.storage.path(source["storage_key"]))
