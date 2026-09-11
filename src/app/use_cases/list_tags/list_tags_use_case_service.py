from dataclasses import asdict
from typing import Callable

from app.entities.tag.services.list_tags.dtos.list_tags_dto_in import ListTagsDtoIn as EntityDtoIn
from app.entities.tag.services.list_tags.list_tags_service import ListTagsService
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.services.chat.lock_chat_owner_service import LockChatOwnerService
from app.use_cases.list_tags.dtos.list_tags_dto_in import ListTagsDtoIn
from app.use_cases.list_tags.dtos.list_tags_dto_out import ListTagsDtoOut


class ListTagsUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork]):
        self.uow_factory = uow_factory

    def exec(self, dto: ListTagsDtoIn) -> ListTagsDtoOut:
        if "tag.read" not in dto.actor.permissions:
            raise AuthorizationError()
        with self.uow_factory() as uow:
            LockChatOwnerService(uow.jobs).exec(dto.actor.office_id, dto.actor.user_customer_id)
            result = (
                ListTagsService(uow.tags)
                .exec(
                    EntityDtoIn(
                        dto.actor.office_id, dto.actor.user_id, dto.actor.user_customer_id, dto.limit, dto.offset
                    )
                )
                .data
            )
            uow.commit()
            return ListTagsDtoOut([asdict(item) for item in result])
