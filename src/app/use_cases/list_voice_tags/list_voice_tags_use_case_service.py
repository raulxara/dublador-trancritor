from dataclasses import asdict
from typing import Callable

from app.entities.tag.services.list_voice_tags.dtos.list_voice_tags_dto_in import ListVoiceTagsDtoIn as EntityDtoIn
from app.entities.tag.services.list_voice_tags.list_voice_tags_service import ListVoiceTagsService
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.services.chat.lock_chat_owner_service import LockChatOwnerService
from app.use_cases.list_voice_tags.dtos.list_voice_tags_dto_in import ListVoiceTagsDtoIn
from app.use_cases.list_voice_tags.dtos.list_voice_tags_dto_out import ListVoiceTagsDtoOut


class ListVoiceTagsUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork]):
        self.uow_factory = uow_factory

    def exec(self, dto: ListVoiceTagsDtoIn) -> ListVoiceTagsDtoOut:
        if "voice.read" not in dto.actor.permissions:
            raise AuthorizationError()
        with self.uow_factory() as uow:
            LockChatOwnerService(uow.jobs).exec(dto.actor.office_id, dto.actor.user_customer_id)
            result = (
                ListVoiceTagsService(uow.tags)
                .exec(EntityDtoIn(dto.actor.office_id, dto.actor.user_id, dto.actor.user_customer_id, dto.voice_id))
                .data
            )
            uow.commit()
            return ListVoiceTagsDtoOut([asdict(item) for item in result])
