from fastapi import Depends, Request

from app.http.dependencies.require_permission import RequirePermission
from app.http.schemas.update_dubbing_chat_input import UpdateDubbingChatInput
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.update_dubbing_chat.dtos.update_dubbing_chat_dto_in import UpdateDubbingChatDtoIn


def update_dubbing_chat(
    chat_id: str,
    body: UpdateDubbingChatInput,
    request: Request,
    actor: AuthorizedActor = Depends(RequirePermission("dubbing_chat.update")),
):
    dto = UpdateDubbingChatDtoIn(actor, chat_id, **body.model_dump())
    result = request.app.state.container.update_dubbing_chat.exec(dto)
    return {"success": True, "data": result.data}
