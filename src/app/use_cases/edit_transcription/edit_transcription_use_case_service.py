from dataclasses import asdict
from typing import Callable

from app.entities.dubbing_chat.services.require_owned_chat.dtos.require_owned_chat_dto_in import RequireOwnedChatDtoIn
from app.entities.dubbing_chat.services.require_owned_chat.require_owned_chat_service import RequireOwnedChatService
from app.entities.dubbing_job.services.require_owned_job.dtos.require_owned_job_dto_in import RequireOwnedJobDtoIn
from app.entities.dubbing_job.services.require_owned_job.require_owned_job_service import RequireOwnedJobService
from app.entities.transcription.services.edit_transcription.dtos.edit_transcription_dto_in import (
    EditTranscriptionDtoIn as EntityDtoIn,
)
from app.entities.transcription.services.edit_transcription.edit_transcription_service import EditTranscriptionService
from app.exceptions.authorization_error import AuthorizationError
from app.exceptions.conflict_error import ConflictError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.services.chat.lock_chat_owner_service import LockChatOwnerService
from app.use_cases.edit_transcription.dtos.edit_transcription_dto_in import EditTranscriptionDtoIn
from app.use_cases.edit_transcription.dtos.edit_transcription_dto_out import EditTranscriptionDtoOut


class EditTranscriptionUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork]):
        self.uow_factory = uow_factory

    def exec(self, dto: EditTranscriptionDtoIn) -> EditTranscriptionDtoOut:
        if "transcription.update" not in dto.actor.permissions:
            raise AuthorizationError()
        with self.uow_factory() as uow:
            LockChatOwnerService(uow.jobs).exec(dto.actor.office_id, dto.actor.user_customer_id)
            job = (
                RequireOwnedJobService(uow.jobs)
                .exec(RequireOwnedJobDtoIn(dto.actor.office_id, dto.actor.user_customer_id, dto.job_id))
                .data
            )
            RequireOwnedChatService(uow.chats).exec(
                RequireOwnedChatDtoIn(dto.actor.office_id, dto.actor.user_customer_id, job.chat_id)
            )
            if job.processing_state != "completed":
                raise ConflictError()
            result = (
                EditTranscriptionService(uow.transcriptions)
                .exec(
                    EntityDtoIn(
                        dto.actor.office_id,
                        dto.actor.user_id,
                        dto.actor.user_customer_id,
                        dto.job_id,
                        dto.base_transcription_id,
                        dto.text,
                    )
                )
                .data
            )
            uow.commit()
            return EditTranscriptionDtoOut(asdict(result))
