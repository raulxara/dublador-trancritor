from fastapi import Depends, Query, Request

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.list_dubbing_jobs.dtos.list_dubbing_jobs_dto_in import ListDubbingJobsDtoIn


def list_dubbing_jobs(
    chat_id: str,
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    actor: AuthorizedActor = Depends(RequirePermission("dubbing.read")),
):
    dto = ListDubbingJobsDtoIn(actor, chat_id, limit, offset)
    result = request.app.state.container.list_dubbing_jobs.exec(dto)
    return {"success": True, "data": result.data, "limit": limit, "offset": offset}
