from dataclasses import asdict
from typing import Callable

from app.entities.dubbing_chat.services.require_owned_chat.dtos.require_owned_chat_dto_in import RequireOwnedChatDtoIn
from app.entities.dubbing_chat.services.require_owned_chat.require_owned_chat_service import RequireOwnedChatService
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.use_cases.get_dubbing_chat.dtos.get_dubbing_chat_dto_in import GetDubbingChatDtoIn
from app.use_cases.get_dubbing_chat.dtos.get_dubbing_chat_dto_out import GetDubbingChatDtoOut


class GetDubbingChatUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork], storage: IPrivateMediaStorage):
        self.uow_factory, self.storage = uow_factory, storage

    def exec(self, dto: GetDubbingChatDtoIn) -> GetDubbingChatDtoOut:
        if "dubbing_chat.read" not in dto.actor.permissions:
            raise AuthorizationError()

        with self.uow_factory() as uow:
            chat = (
                RequireOwnedChatService(uow.chats)
                .exec(RequireOwnedChatDtoIn(dto.actor.office_id, dto.actor.user_customer_id, dto.chat_id, True))
                .data
            )
            return GetDubbingChatDtoOut(asdict(chat))
