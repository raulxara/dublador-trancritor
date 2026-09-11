from dataclasses import asdict
from typing import Callable

from app.entities.dubbing_chat.services.list_owned_chats.dtos.list_owned_chats_dto_in import ListOwnedChatsDtoIn
from app.entities.dubbing_chat.services.list_owned_chats.list_owned_chats_service import ListOwnedChatsService
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.use_cases.list_dubbing_chats.dtos.list_dubbing_chats_dto_in import ListDubbingChatsDtoIn
from app.use_cases.list_dubbing_chats.dtos.list_dubbing_chats_dto_out import ListDubbingChatsDtoOut


class ListDubbingChatsUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork], storage: IPrivateMediaStorage):
        self.uow_factory, self.storage = uow_factory, storage

    def exec(self, dto: ListDubbingChatsDtoIn) -> ListDubbingChatsDtoOut:
        if "dubbing_chat.read" not in dto.actor.permissions:
            raise AuthorizationError()

        with self.uow_factory() as uow:
            rows = (
                ListOwnedChatsService(uow.chats)
                .exec(ListOwnedChatsDtoIn(dto.actor.office_id, dto.actor.user_customer_id, dto.limit, dto.offset))
                .data
            )
            return ListDubbingChatsDtoOut([asdict(row) for row in rows])
