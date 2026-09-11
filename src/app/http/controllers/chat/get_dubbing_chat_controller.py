from fastapi import Depends, Request

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.get_dubbing_chat.dtos.get_dubbing_chat_dto_in import GetDubbingChatDtoIn


def get_dubbing_chat(
    chat_id: str, request: Request, actor: AuthorizedActor = Depends(RequirePermission("dubbing_chat.read"))
):
    dto = GetDubbingChatDtoIn(actor, chat_id)
    result = request.app.state.container.get_dubbing_chat.exec(dto)
    return {"success": True, "data": result.data}
