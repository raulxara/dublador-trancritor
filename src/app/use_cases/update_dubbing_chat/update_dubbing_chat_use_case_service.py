from dataclasses import asdict, replace
from typing import Callable

from app.entities.dubbing_chat.services.require_owned_chat.dtos.require_owned_chat_dto_in import RequireOwnedChatDtoIn
from app.entities.dubbing_chat.services.require_owned_chat.require_owned_chat_service import RequireOwnedChatService
from app.entities.dubbing_chat.services.save_chat.dtos.save_chat_dto_in import SaveChatDtoIn
from app.entities.dubbing_chat.services.save_chat.save_chat_service import SaveChatService
from app.exceptions.authorization_error import AuthorizationError
from app.exceptions.invalid_input_error import InvalidInputError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.services.chat.ensure_chat_idle_service import EnsureChatIdleService
from app.services.chat.lock_chat_owner_service import LockChatOwnerService
from app.services.chat.ready_chat_source_service import ReadyChatSourceService
from app.use_cases.update_dubbing_chat.dtos.update_dubbing_chat_dto_in import UpdateDubbingChatDtoIn
from app.use_cases.update_dubbing_chat.dtos.update_dubbing_chat_dto_out import UpdateDubbingChatDtoOut


class UpdateDubbingChatUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork], storage: IPrivateMediaStorage):
        self.uow_factory, self.storage = uow_factory, storage

    def exec(self, dto: UpdateDubbingChatDtoIn) -> UpdateDubbingChatDtoOut:
        if "dubbing_chat.update" not in dto.actor.permissions:
            raise AuthorizationError()

        if dto.status not in ("active", "inactive"):
            raise InvalidInputError()
        with self.uow_factory() as uow:
            LockChatOwnerService(uow.jobs).exec(dto.actor.office_id, dto.actor.user_customer_id)
            chat = (
                RequireOwnedChatService(uow.chats)
                .exec(RequireOwnedChatDtoIn(dto.actor.office_id, dto.actor.user_customer_id, dto.chat_id, True))
                .data
            )
            if dto.selected_voice_id is not None:
                ReadyChatSourceService(uow.sources, self.storage).sample(dto.actor.office_id, dto.selected_voice_id)
            if dto.status == "inactive":
                EnsureChatIdleService(uow.jobs).exec(dto.actor.office_id, dto.chat_id)
            chat = replace(chat, title=dto.title, selected_voice_id=dto.selected_voice_id, status=dto.status)
            SaveChatService(uow.chats).exec(SaveChatDtoIn(chat, False))
            uow.commit()
            return UpdateDubbingChatDtoOut(asdict(chat))
