from app.entities.dubbing_chat.i_dubbing_chat_repository import IChatRepository
from app.entities.dubbing_chat.services.save_chat.dtos.save_chat_dto_in import SaveChatDtoIn
from app.entities.dubbing_chat.services.save_chat.dtos.save_chat_dto_out import SaveChatDtoOut


class SaveChatService:
    def __init__(self, repository: IChatRepository):
        self.repository = repository

    def exec(self, dto: SaveChatDtoIn) -> SaveChatDtoOut:
        self.repository.save(dto.chat, dto.create)
        return SaveChatDtoOut(dto.chat)
