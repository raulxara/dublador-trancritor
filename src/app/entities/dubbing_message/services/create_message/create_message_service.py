from app.entities.dubbing_message.i_dubbing_message_repository import IMessageRepository
from app.entities.dubbing_message.services.create_message.dtos.create_message_dto_in import CreateMessageDtoIn
from app.entities.dubbing_message.services.create_message.dtos.create_message_dto_out import CreateMessageDtoOut


class CreateMessageService:
    def __init__(self, repository: IMessageRepository):
        self.repository = repository

    def exec(self, dto: CreateMessageDtoIn) -> CreateMessageDtoOut:
        self.repository.create(dto.message)
        return CreateMessageDtoOut(dto.message)
