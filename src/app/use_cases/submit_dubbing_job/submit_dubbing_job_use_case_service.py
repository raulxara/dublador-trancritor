from typing import Callable
from uuid import uuid4

from app.entities.dubbing_chat.services.require_owned_chat.dtos.require_owned_chat_dto_in import RequireOwnedChatDtoIn
from app.entities.dubbing_chat.services.require_owned_chat.require_owned_chat_service import RequireOwnedChatService
from app.entities.dubbing_job.dubbing_job_entity import DubbingJobEntity
from app.entities.dubbing_job.services.create_job.create_job_service import CreateJobService
from app.entities.dubbing_job.services.create_job.dtos.create_job_dto_in import CreateJobDtoIn
from app.entities.dubbing_job.services.find_idempotent_job.dtos.find_idempotent_job_dto_in import FindIdempotentJobDtoIn
from app.entities.dubbing_job.services.find_idempotent_job.find_idempotent_job_service import FindIdempotentJobService
from app.entities.dubbing_message.dubbing_message_entity import DubbingMessageEntity
from app.entities.dubbing_message.services.create_message.create_message_service import CreateMessageService
from app.entities.dubbing_message.services.create_message.dtos.create_message_dto_in import CreateMessageDtoIn
from app.entities.dubbing_message.services.require_source_message.dtos.require_source_message_dto_in import (
    RequireSourceMessageDtoIn,
)
from app.entities.dubbing_message.services.require_source_message.require_source_message_service import (
    RequireSourceMessageService,
)
from app.exceptions.authorization_error import AuthorizationError
from app.interfaces.i_chat_unit_of_work import IChatUnitOfWork
from app.interfaces.i_private_media_storage import IPrivateMediaStorage
from app.services.chat.hash_dubbing_request_service import HashDubbingRequestService
from app.services.chat.job_view_service import JobViewService
from app.services.chat.ready_chat_source_service import ReadyChatSourceService
from app.use_cases.submit_dubbing_job.dtos.submit_dubbing_job_dto_in import SubmitDubbingJobDtoIn
from app.use_cases.submit_dubbing_job.dtos.submit_dubbing_job_dto_out import SubmitDubbingJobDtoOut


class SubmitDubbingJobUseCaseService:
    def __init__(self, uow_factory: Callable[[], IChatUnitOfWork], storage: IPrivateMediaStorage):
        self.uow_factory, self.storage = uow_factory, storage

    def exec(self, dto: SubmitDubbingJobDtoIn) -> SubmitDubbingJobDtoOut:
        if "dubbing.generate" not in dto.actor.permissions:
            raise AuthorizationError()

        request_hash = HashDubbingRequestService().exec(dto)
        with self.uow_factory() as uow:
            old = (
                FindIdempotentJobService(uow.jobs)
                .exec(
                    FindIdempotentJobDtoIn(
                        dto.actor.office_id, dto.actor.user_customer_id, dto.idempotency_key, request_hash
                    )
                )
                .data
            )
            chat = (
                RequireOwnedChatService(uow.chats)
                .exec(RequireOwnedChatDtoIn(dto.actor.office_id, dto.actor.user_customer_id, dto.chat_id))
                .data
            )
            if old is not None:
                return SubmitDubbingJobDtoOut(JobViewService.exec(old))
            sources = ReadyChatSourceService(uow.sources, self.storage)
            sample = None
            language = dto.target_language_id
            if dto.operation != "transcribe":
                sample = sources.sample(dto.actor.office_id, chat.selected_voice_id)
                language = language or sample["language_id"]
            sources.language(language)
            input_file = None
            message_id = dto.input_message_id
            text = dto.input_text.strip() if dto.input_text is not None else None
            if dto.operation == "text_to_speech":
                message = DubbingMessageEntity(
                    str(uuid4()), dto.actor.office_id, dto.chat_id, dto.actor.user_customer_id, "user", "text", text
                )
                CreateMessageService(uow.messages).exec(CreateMessageDtoIn(message))
                message_id = message.unique_id
            else:
                RequireSourceMessageService(uow.messages).exec(
                    RequireSourceMessageDtoIn(dto.actor.office_id, dto.chat_id, dto.actor.user_customer_id, message_id)
                )
                input_file = sources.input(dto.actor.office_id, message_id)["file_id"]
            job = DubbingJobEntity(
                str(uuid4()),
                dto.actor.office_id,
                dto.actor.user_customer_id,
                dto.chat_id,
                message_id,
                dto.operation,
                text,
                input_file,
                sample["sample_id"] if sample else None,
                language,
                dto.speed,
                dto.pitch_semitones,
                dto.preserve_timing,
                '{"format":"txt"}' if dto.operation == "transcribe" else '{"format":"wav"}',
                dto.idempotency_key,
                request_hash,
            )
            CreateJobService(uow.jobs).exec(CreateJobDtoIn(job))
            uow.commit()
            return SubmitDubbingJobDtoOut(JobViewService.exec(job))
