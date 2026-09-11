from typing import Callable

from app.entities.dubbing_chat.services.require_owned_chat.dtos.require_owned_chat_dto_in import RequireOwnedChatDtoIn
from app.entities.dubbing_chat.services.require_owned_chat.require_owned_chat_service import RequireOwnedChatService
from app.entities.dubbing_job.services.list_chat_jobs.dtos.list_chat_jobs_dto_in import ListChatJobsDtoIn
from app.entities.dubbing_job.services.list_chat_jobs.list_chat_jobs_service import ListChatJobsService
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.services.chat.job_view_service import JobViewService
from app.use_cases.list_dubbing_jobs.dtos.list_dubbing_jobs_dto_in import ListDubbingJobsDtoIn
from app.use_cases.list_dubbing_jobs.dtos.list_dubbing_jobs_dto_out import ListDubbingJobsDtoOut


class ListDubbingJobsUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork], storage: IPrivateMediaStorage):
        self.uow_factory, self.storage = uow_factory, storage

    def exec(self, dto: ListDubbingJobsDtoIn) -> ListDubbingJobsDtoOut:
        if "dubbing.read" not in dto.actor.permissions:
            raise AuthorizationError()

        with self.uow_factory() as uow:
            RequireOwnedChatService(uow.chats).exec(
                RequireOwnedChatDtoIn(dto.actor.office_id, dto.actor.user_customer_id, dto.chat_id)
            ).data
            rows = (
                ListChatJobsService(uow.jobs)
                .exec(
                    ListChatJobsDtoIn(
                        dto.actor.office_id, dto.actor.user_customer_id, dto.chat_id, dto.limit, dto.offset
                    )
                )
                .data
            )
            return ListDubbingJobsDtoOut([JobViewService.exec(row) for row in rows])
