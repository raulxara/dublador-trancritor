from dataclasses import asdict
from typing import Callable

from app.entities.project_audio_link.services.list_project_audio_links.dtos.list_project_audio_links_dto_in import (
    ListProjectAudioLinksDtoIn as EntityDtoIn,
)
from app.entities.project_audio_link.services.list_project_audio_links.list_project_audio_links_service import (
    ListProjectAudioLinksService,
)
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.services.chat.lock_chat_owner_service import LockChatOwnerService
from app.use_cases.list_project_audio_links.dtos.list_project_audio_links_dto_in import ListProjectAudioLinksDtoIn
from app.use_cases.list_project_audio_links.dtos.list_project_audio_links_dto_out import ListProjectAudioLinksDtoOut


class ListProjectAudioLinksUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork]):
        self.uow_factory = uow_factory

    def exec(self, dto: ListProjectAudioLinksDtoIn) -> ListProjectAudioLinksDtoOut:
        if "project_audio.read" not in dto.actor.permissions:
            raise AuthorizationError()
        with self.uow_factory() as uow:
            LockChatOwnerService(uow.jobs).exec(dto.actor.office_id, dto.actor.user_customer_id)
            result = (
                ListProjectAudioLinksService(uow.project_links)
                .exec(
                    EntityDtoIn(
                        dto.actor.office_id,
                        dto.actor.user_id,
                        dto.actor.user_customer_id,
                        dto.project_id,
                        dto.limit,
                        dto.offset,
                    )
                )
                .data
            )
            uow.commit()
            return ListProjectAudioLinksDtoOut([asdict(item) for item in result])
