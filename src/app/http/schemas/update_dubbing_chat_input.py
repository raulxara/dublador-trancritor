from typing import Literal

from app.http.schemas.create_dubbing_chat_input import CreateDubbingChatInput


class UpdateDubbingChatInput(CreateDubbingChatInput):
    status: Literal["active", "inactive"] = "active"
