from dataclasses import asdict
from typing import Callable

from app.entities.tag.services.update_tag.dtos.update_tag_dto_in import UpdateTagDtoIn as EntityDtoIn
from app.entities.tag.services.update_tag.update_tag_service import UpdateTagService
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.services.chat.lock_chat_owner_service import LockChatOwnerService
from app.use_cases.update_tag.dtos.update_tag_dto_in import UpdateTagDtoIn
from app.use_cases.update_tag.dtos.update_tag_dto_out import UpdateTagDtoOut


class UpdateTagUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork]):
        self.uow_factory = uow_factory

    def exec(self, dto: UpdateTagDtoIn) -> UpdateTagDtoOut:
        if "tag.update" not in dto.actor.permissions:
            raise AuthorizationError()
        with self.uow_factory() as uow:
            LockChatOwnerService(uow.jobs).exec(dto.actor.office_id, dto.actor.user_customer_id)
            result = (
                UpdateTagService(uow.tags)
                .exec(
                    EntityDtoIn(
                        dto.actor.office_id,
                        dto.actor.user_id,
                        dto.actor.user_customer_id,
                        dto.tag_id,
                        dto.name,
                        dto.slug,
                        dto.status,
                    )
                )
                .data
            )
            uow.commit()
            return UpdateTagDtoOut(asdict(result))
