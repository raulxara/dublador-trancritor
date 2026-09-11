from dataclasses import asdict
from typing import Callable

from app.entities.dubbing_chat.services.require_owned_chat.dtos.require_owned_chat_dto_in import RequireOwnedChatDtoIn
from app.entities.dubbing_chat.services.require_owned_chat.require_owned_chat_service import RequireOwnedChatService
from app.entities.dubbing_message.services.list_messages.dtos.list_messages_dto_in import ListMessagesDtoIn
from app.entities.dubbing_message.services.list_messages.list_messages_service import ListMessagesService
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.use_cases.list_dubbing_messages.dtos.list_dubbing_messages_dto_in import ListDubbingMessagesDtoIn
from app.use_cases.list_dubbing_messages.dtos.list_dubbing_messages_dto_out import ListDubbingMessagesDtoOut


class ListDubbingMessagesUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork], storage: IPrivateMediaStorage):
        self.uow_factory, self.storage = uow_factory, storage

    def exec(self, dto: ListDubbingMessagesDtoIn) -> ListDubbingMessagesDtoOut:
        if "dubbing_chat.read" not in dto.actor.permissions:
            raise AuthorizationError()

        with self.uow_factory() as uow:
            RequireOwnedChatService(uow.chats).exec(
                RequireOwnedChatDtoIn(dto.actor.office_id, dto.actor.user_customer_id, dto.chat_id)
            ).data
            rows = (
                ListMessagesService(uow.messages)
                .exec(ListMessagesDtoIn(dto.actor.office_id, dto.chat_id, dto.limit, dto.offset))
                .data
            )
            return ListDubbingMessagesDtoOut([asdict(row) for row in rows])
