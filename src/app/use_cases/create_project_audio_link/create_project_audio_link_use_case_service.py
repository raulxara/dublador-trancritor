from dataclasses import asdict
from typing import Callable

from app.entities.project_audio_link.services.create_project_audio_link.create_project_audio_link_service import (
    CreateProjectAudioLinkService,
)
from app.entities.project_audio_link.services.create_project_audio_link.dtos.create_project_audio_link_dto_in import (
    CreateProjectAudioLinkDtoIn as EntityDtoIn,
)
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.services.chat.lock_chat_owner_service import LockChatOwnerService
from app.use_cases.create_project_audio_link.dtos.create_project_audio_link_dto_in import CreateProjectAudioLinkDtoIn
from app.use_cases.create_project_audio_link.dtos.create_project_audio_link_dto_out import CreateProjectAudioLinkDtoOut


class CreateProjectAudioLinkUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork]):
        self.uow_factory = uow_factory

    def exec(self, dto: CreateProjectAudioLinkDtoIn) -> CreateProjectAudioLinkDtoOut:
        if "project_audio.update" not in dto.actor.permissions:
            raise AuthorizationError()
        with self.uow_factory() as uow:
            LockChatOwnerService(uow.jobs).exec(dto.actor.office_id, dto.actor.user_customer_id)
            result = (
                CreateProjectAudioLinkService(uow.project_links)
                .exec(
                    EntityDtoIn(
                        dto.actor.office_id,
                        dto.actor.user_id,
                        dto.actor.user_customer_id,
                        dto.project_id,
                        dto.output_id,
                    )
                )
                .data
            )
            uow.commit()
            return CreateProjectAudioLinkDtoOut(asdict(result))
