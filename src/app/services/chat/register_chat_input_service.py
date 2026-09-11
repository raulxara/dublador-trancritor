from app.interfaces.i_chat_source_repository import IChatSourceRepository


class RegisterChatInputService:
    def __init__(self, repository: IChatSourceRepository):
        self.repository = repository

    def exec(self, office_id: str, user_id: str, message_id: str, metadata: dict) -> None:
        self.repository.create_input(office_id, user_id, message_id, metadata)
