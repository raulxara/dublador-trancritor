from typing import Callable

from app.entities.dubbing_chat.services.require_owned_chat.dtos.require_owned_chat_dto_in import RequireOwnedChatDtoIn
from app.entities.dubbing_chat.services.require_owned_chat.require_owned_chat_service import RequireOwnedChatService
from app.entities.dubbing_job.services.list_job_outputs.dtos.list_job_outputs_dto_in import ListJobOutputsDtoIn
from app.entities.dubbing_job.services.list_job_outputs.list_job_outputs_service import ListJobOutputsService
from app.entities.dubbing_job.services.require_owned_job.dtos.require_owned_job_dto_in import RequireOwnedJobDtoIn
from app.entities.dubbing_job.services.require_owned_job.require_owned_job_service import RequireOwnedJobService
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.use_cases.list_dubbing_outputs.dtos.list_dubbing_outputs_dto_in import ListDubbingOutputsDtoIn
from app.use_cases.list_dubbing_outputs.dtos.list_dubbing_outputs_dto_out import ListDubbingOutputsDtoOut


class ListDubbingOutputsUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork], storage: IPrivateMediaStorage):
        self.uow_factory, self.storage = uow_factory, storage

    def exec(self, dto: ListDubbingOutputsDtoIn) -> ListDubbingOutputsDtoOut:
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
            outputs = ListJobOutputsService(uow.outputs).exec(ListJobOutputsDtoIn(dto.actor.office_id, dto.job_id)).data
            return ListDubbingOutputsDtoOut(
                [
                    dict(
                        unique_id=item.unique_id,
                        format=item.format,
                        purpose=item.purpose,
                        size_bytes=item.size_bytes,
                        duration_ms=item.duration_ms,
                    )
                    for item in outputs
                ]
            )
