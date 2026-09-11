from typing import Callable

from app.entities.dubbing_chat.services.require_owned_chat.dtos.require_owned_chat_dto_in import RequireOwnedChatDtoIn
from app.entities.dubbing_chat.services.require_owned_chat.require_owned_chat_service import RequireOwnedChatService
from app.entities.dubbing_job.services.require_owned_job.dtos.require_owned_job_dto_in import RequireOwnedJobDtoIn
from app.entities.dubbing_job.services.require_owned_job.require_owned_job_service import RequireOwnedJobService
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.services.chat.job_view_service import JobViewService
from app.use_cases.get_dubbing_job.dtos.get_dubbing_job_dto_in import GetDubbingJobDtoIn
from app.use_cases.get_dubbing_job.dtos.get_dubbing_job_dto_out import GetDubbingJobDtoOut


class GetDubbingJobUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork], storage: IPrivateMediaStorage):
        self.uow_factory, self.storage = uow_factory, storage

    def exec(self, dto: GetDubbingJobDtoIn) -> GetDubbingJobDtoOut:
        if "dubbing.read" not in dto.actor.permissions:
            raise AuthorizationError()

        with self.uow_factory() as uow:
            job = (
                RequireOwnedJobService(uow.jobs)
                .exec(RequireOwnedJobDtoIn(dto.actor.office_id, dto.actor.user_customer_id, dto.job_id))
                .data
            )
            RequireOwnedChatService(uow.chats).exec(
                RequireOwnedChatDtoIn(dto.actor.office_id, dto.actor.user_customer_id, job.chat_id)
            )
            return GetDubbingJobDtoOut(JobViewService.exec(job))
