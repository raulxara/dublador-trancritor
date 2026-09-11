from app.entities.dubbing_chat.i_dubbing_chat_repository import IChatRepository
from app.entities.dubbing_chat.services.require_owned_chat.dtos.require_owned_chat_dto_in import RequireOwnedChatDtoIn
from app.entities.dubbing_chat.services.require_owned_chat.dtos.require_owned_chat_dto_out import RequireOwnedChatDtoOut
from app.exceptions.resource_not_found_error import ResourceNotFoundError


class RequireOwnedChatService:
    def __init__(self, repository: IChatRepository):
        self.repository = repository

    def exec(self, dto: RequireOwnedChatDtoIn) -> RequireOwnedChatDtoOut:
        chat = self.repository.find(dto.office_id, dto.chat_id, lock=True)
        if (
            chat is None
            or chat.office_id != dto.office_id
            or chat.user_customer_id != dto.owner_id
            or (chat.status != "active" and not dto.allow_inactive)
        ):
            raise ResourceNotFoundError()
        return RequireOwnedChatDtoOut(chat)
