from fastapi import Depends, Query, Request

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.list_transcriptions.dtos.list_transcriptions_dto_in import ListTranscriptionsDtoIn


def list_transcriptions(
    request: Request,
    job_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    actor: AuthorizedActor = Depends(RequirePermission("dubbing.read")),
):
    result = request.app.state.container.list_transcriptions.exec(ListTranscriptionsDtoIn(actor, job_id, limit, offset))
    return {"success": True, "data": result.data}
