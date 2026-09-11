from typing import Callable

from app.entities.dubbing_chat.services.require_owned_chat.dtos.require_owned_chat_dto_in import RequireOwnedChatDtoIn
from app.entities.dubbing_chat.services.require_owned_chat.require_owned_chat_service import RequireOwnedChatService
from app.entities.dubbing_job.services.list_job_outputs.dtos.list_job_outputs_dto_in import ListJobOutputsDtoIn
from app.entities.dubbing_job.services.list_job_outputs.list_job_outputs_service import ListJobOutputsService
from app.entities.dubbing_job.services.require_owned_job.dtos.require_owned_job_dto_in import RequireOwnedJobDtoIn
from app.entities.dubbing_job.services.require_owned_job.require_owned_job_service import RequireOwnedJobService
from app.exceptions.authorization_error import AuthorizationError
from app.exceptions.resource_not_found_error import ResourceNotFoundError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.use_cases.download_dubbing_output.dtos.download_dubbing_output_dto_in import DownloadDubbingOutputDtoIn
from app.use_cases.download_dubbing_output.dtos.download_dubbing_output_dto_out import DownloadDubbingOutputDtoOut


class DownloadDubbingOutputUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork], storage: IPrivateMediaStorage):
        self.uow_factory, self.storage = uow_factory, storage

    def exec(self, dto: DownloadDubbingOutputDtoIn) -> DownloadDubbingOutputDtoOut:
        if "dubbing.download" not in dto.actor.permissions:
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
            output = next((item for item in outputs if item.unique_id == dto.output_id), None)
            if output is None:
                raise ResourceNotFoundError()
            return DownloadDubbingOutputDtoOut(self.storage.path(output.storage_key), output.format)
