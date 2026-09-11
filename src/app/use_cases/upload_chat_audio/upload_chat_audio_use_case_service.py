from dataclasses import asdict
from typing import Callable
from uuid import uuid4

from app.entities.dubbing_chat.services.require_owned_chat.dtos.require_owned_chat_dto_in import RequireOwnedChatDtoIn
from app.entities.dubbing_chat.services.require_owned_chat.require_owned_chat_service import RequireOwnedChatService
from app.entities.dubbing_message.dubbing_message_entity import DubbingMessageEntity
from app.entities.dubbing_message.services.create_message.create_message_service import CreateMessageService
from app.entities.dubbing_message.services.create_message.dtos.create_message_dto_in import CreateMessageDtoIn
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.services.chat.lock_chat_owner_service import LockChatOwnerService
from app.services.chat.prepare_chat_audio_service import PrepareChatAudioService
from app.services.chat.register_chat_input_service import RegisterChatInputService
from app.use_cases.upload_chat_audio.dtos.upload_chat_audio_dto_in import UploadChatAudioDtoIn
from app.use_cases.upload_chat_audio.dtos.upload_chat_audio_dto_out import UploadChatAudioDtoOut


class UploadChatAudioUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork], storage: IPrivateMediaStorage):
        self.uow_factory, self.storage = uow_factory, storage

    def exec(self, dto: UploadChatAudioDtoIn) -> UploadChatAudioDtoOut:
        if "dubbing.generate" not in dto.actor.permissions:
            raise AuthorizationError()

        with self.uow_factory() as uow:
            LockChatOwnerService(uow.jobs).exec(dto.actor.office_id, dto.actor.user_customer_id)
            RequireOwnedChatService(uow.chats).exec(
                RequireOwnedChatDtoIn(dto.actor.office_id, dto.actor.user_customer_id, dto.chat_id)
            ).data
        metadata = PrepareChatAudioService(self.storage).exec(dto.actor.office_id, dto.content)
        with self.uow_factory() as uow:
            LockChatOwnerService(uow.jobs).exec(dto.actor.office_id, dto.actor.user_customer_id)
            RequireOwnedChatService(uow.chats).exec(
                RequireOwnedChatDtoIn(dto.actor.office_id, dto.actor.user_customer_id, dto.chat_id)
            ).data
            message = DubbingMessageEntity(
                str(uuid4()), dto.actor.office_id, dto.chat_id, dto.actor.user_customer_id, "user", "audio", None
            )
            CreateMessageService(uow.messages).exec(CreateMessageDtoIn(message))
            RegisterChatInputService(uow.sources).exec(
                dto.actor.office_id, dto.actor.user_id, message.unique_id, metadata
            )
            uow.commit()
            return UploadChatAudioDtoOut(asdict(message))
