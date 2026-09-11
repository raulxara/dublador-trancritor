from dataclasses import asdict
from typing import Callable

from app.entities.project_audio_link.services.unlink_project_audio.dtos.unlink_project_audio_dto_in import (
    UnlinkProjectAudioDtoIn as EntityDtoIn,
)
from app.entities.project_audio_link.services.unlink_project_audio.unlink_project_audio_service import (
    UnlinkProjectAudioService,
)
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.services.chat.lock_chat_owner_service import LockChatOwnerService
from app.use_cases.unlink_project_audio.dtos.unlink_project_audio_dto_in import (
    UnlinkProjectAudioDtoIn,
)
from app.use_cases.unlink_project_audio.dtos.unlink_project_audio_dto_out import (
    UnlinkProjectAudioDtoOut,
)


class UnlinkProjectAudioUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork]):
        self.uow_factory = uow_factory

    def exec(self, dto: UnlinkProjectAudioDtoIn) -> UnlinkProjectAudioDtoOut:
        if "project_audio.update" not in dto.actor.permissions:
            raise AuthorizationError()
        with self.uow_factory() as uow:
            LockChatOwnerService(uow.jobs).exec(dto.actor.office_id, dto.actor.user_customer_id)
            result = (
                UnlinkProjectAudioService(uow.project_links)
                .exec(EntityDtoIn(dto.actor.office_id, dto.actor.user_id, dto.actor.user_customer_id, dto.link_id))
                .data
            )
            uow.commit()
            return UnlinkProjectAudioDtoOut(asdict(result))
