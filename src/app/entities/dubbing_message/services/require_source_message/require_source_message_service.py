from app.entities.dubbing_message.i_dubbing_message_repository import IMessageRepository
from app.entities.dubbing_message.services.require_source_message.dtos.require_source_message_dto_in import (
    RequireSourceMessageDtoIn,
)
from app.entities.dubbing_message.services.require_source_message.dtos.require_source_message_dto_out import (
    RequireSourceMessageDtoOut,
)
from app.exceptions.resource_not_found_error import ResourceNotFoundError


class RequireSourceMessageService:
    def __init__(self, repository: IMessageRepository):
        self.repository = repository

    def exec(self, dto: RequireSourceMessageDtoIn) -> RequireSourceMessageDtoOut:
        message = self.repository.find(dto.office_id, dto.chat_id, dto.message_id)
        if (
            message is None
            or message.user_customer_id != dto.owner_id
            or message.status != "active"
            or message.role != "user"
            or message.message_type != "audio"
        ):
            raise ResourceNotFoundError()
        return RequireSourceMessageDtoOut(message)
