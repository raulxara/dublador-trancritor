from dataclasses import asdict
from typing import Callable

from app.entities.tag.services.create_tag.create_tag_service import CreateTagService
from app.entities.tag.services.create_tag.dtos.create_tag_dto_in import CreateTagDtoIn as EntityDtoIn
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.services.chat.lock_chat_owner_service import LockChatOwnerService
from app.use_cases.create_tag.dtos.create_tag_dto_in import CreateTagDtoIn
from app.use_cases.create_tag.dtos.create_tag_dto_out import CreateTagDtoOut


class CreateTagUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork]):
        self.uow_factory = uow_factory

    def exec(self, dto: CreateTagDtoIn) -> CreateTagDtoOut:
        if "tag.update" not in dto.actor.permissions:
            raise AuthorizationError()
        with self.uow_factory() as uow:
            LockChatOwnerService(uow.jobs).exec(dto.actor.office_id, dto.actor.user_customer_id)
            result = (
                CreateTagService(uow.tags)
                .exec(
                    EntityDtoIn(dto.actor.office_id, dto.actor.user_id, dto.actor.user_customer_id, dto.name, dto.slug)
                )
                .data
            )
            uow.commit()
            return CreateTagDtoOut(asdict(result))
