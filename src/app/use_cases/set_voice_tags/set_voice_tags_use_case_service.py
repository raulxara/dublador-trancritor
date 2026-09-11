from dataclasses import asdict
from typing import Callable

from app.entities.tag.services.set_voice_tags.dtos.set_voice_tags_dto_in import SetVoiceTagsDtoIn as EntityDtoIn
from app.entities.tag.services.set_voice_tags.set_voice_tags_service import SetVoiceTagsService
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.services.chat.lock_chat_owner_service import LockChatOwnerService
from app.use_cases.set_voice_tags.dtos.set_voice_tags_dto_in import SetVoiceTagsDtoIn
from app.use_cases.set_voice_tags.dtos.set_voice_tags_dto_out import SetVoiceTagsDtoOut


class SetVoiceTagsUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork]):
        self.uow_factory = uow_factory

    def exec(self, dto: SetVoiceTagsDtoIn) -> SetVoiceTagsDtoOut:
        if "voice.update" not in dto.actor.permissions:
            raise AuthorizationError()
        with self.uow_factory() as uow:
            LockChatOwnerService(uow.jobs).exec(dto.actor.office_id, dto.actor.user_customer_id)
            result = (
                SetVoiceTagsService(uow.tags)
                .exec(
                    EntityDtoIn(
                        dto.actor.office_id, dto.actor.user_id, dto.actor.user_customer_id, dto.voice_id, dto.tag_ids
                    )
                )
                .data
            )
            uow.commit()
            return SetVoiceTagsDtoOut([asdict(item) for item in result])
