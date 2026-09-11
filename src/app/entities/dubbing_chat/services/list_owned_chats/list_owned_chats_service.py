from app.entities.dubbing_chat.i_dubbing_chat_repository import IChatRepository
from app.entities.dubbing_chat.services.list_owned_chats.dtos.list_owned_chats_dto_in import ListOwnedChatsDtoIn
from app.entities.dubbing_chat.services.list_owned_chats.dtos.list_owned_chats_dto_out import ListOwnedChatsDtoOut


class ListOwnedChatsService:
    def __init__(self, repository: IChatRepository):
        self.repository = repository

    def exec(self, dto: ListOwnedChatsDtoIn) -> ListOwnedChatsDtoOut:
        return ListOwnedChatsDtoOut(self.repository.list(dto.office_id, dto.owner_id, dto.limit, dto.offset))
