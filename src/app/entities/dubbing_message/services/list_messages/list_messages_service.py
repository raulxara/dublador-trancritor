from app.entities.dubbing_message.i_dubbing_message_repository import IMessageRepository
from app.entities.dubbing_message.services.list_messages.dtos.list_messages_dto_in import ListMessagesDtoIn
from app.entities.dubbing_message.services.list_messages.dtos.list_messages_dto_out import ListMessagesDtoOut


class ListMessagesService:
    def __init__(self, repository: IMessageRepository):
        self.repository = repository

    def exec(self, dto: ListMessagesDtoIn) -> ListMessagesDtoOut:
        return ListMessagesDtoOut(self.repository.list(dto.office_id, dto.chat_id, dto.limit, dto.offset))
