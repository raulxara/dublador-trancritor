from fastapi import Depends, Query, Request

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.list_tags.dtos.list_tags_dto_in import ListTagsDtoIn


def list_tags(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    actor: AuthorizedActor = Depends(RequirePermission("tag.read")),
):
    result = request.app.state.container.list_tags.exec(ListTagsDtoIn(actor, limit, offset))
    return {"success": True, "data": result.data}
