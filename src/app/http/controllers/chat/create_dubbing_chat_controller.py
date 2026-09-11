from fastapi import Depends, Request

from app.http.dependencies.require_permission import RequirePermission
from app.http.schemas.create_dubbing_chat_input import CreateDubbingChatInput
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.create_dubbing_chat.dtos.create_dubbing_chat_dto_in import CreateDubbingChatDtoIn


def create_dubbing_chat(
    body: CreateDubbingChatInput,
    request: Request,
    actor: AuthorizedActor = Depends(RequirePermission("dubbing_chat.create")),
):
    dto = CreateDubbingChatDtoIn(actor, **body.model_dump())
    result = request.app.state.container.create_dubbing_chat.exec(dto)
    return {"success": True, "data": result.data}
