from dataclasses import asdict
from typing import Callable
from uuid import uuid4

from app.entities.dubbing_chat.dubbing_chat_entity import DubbingChatEntity
from app.entities.dubbing_chat.services.save_chat.dtos.save_chat_dto_in import SaveChatDtoIn
from app.entities.dubbing_chat.services.save_chat.save_chat_service import SaveChatService
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.services.chat.lock_chat_owner_service import LockChatOwnerService
from app.services.chat.ready_chat_source_service import ReadyChatSourceService
from app.use_cases.create_dubbing_chat.dtos.create_dubbing_chat_dto_in import CreateDubbingChatDtoIn
from app.use_cases.create_dubbing_chat.dtos.create_dubbing_chat_dto_out import CreateDubbingChatDtoOut


class CreateDubbingChatUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork], storage: IPrivateMediaStorage):
        self.uow_factory, self.storage = uow_factory, storage

    def exec(self, dto: CreateDubbingChatDtoIn) -> CreateDubbingChatDtoOut:
        if "dubbing_chat.create" not in dto.actor.permissions:
            raise AuthorizationError()

        with self.uow_factory() as uow:
            LockChatOwnerService(uow.jobs).exec(dto.actor.office_id, dto.actor.user_customer_id)
            if dto.selected_voice_id is not None:
                ReadyChatSourceService(uow.sources, self.storage).sample(dto.actor.office_id, dto.selected_voice_id)
            chat = DubbingChatEntity(
                str(uuid4()), dto.actor.office_id, dto.actor.user_customer_id, dto.title, dto.selected_voice_id
            )
            SaveChatService(uow.chats).exec(SaveChatDtoIn(chat, True))
            uow.commit()
            return CreateDubbingChatDtoOut(asdict(chat))
