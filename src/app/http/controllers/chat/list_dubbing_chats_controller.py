from fastapi import Depends, Query, Request

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.list_dubbing_chats.dtos.list_dubbing_chats_dto_in import ListDubbingChatsDtoIn


def list_dubbing_chats(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    actor: AuthorizedActor = Depends(RequirePermission("dubbing_chat.read")),
):
    dto = ListDubbingChatsDtoIn(actor, limit, offset)
    result = request.app.state.container.list_dubbing_chats.exec(dto)
    return {"success": True, "data": result.data, "limit": limit, "offset": offset}
