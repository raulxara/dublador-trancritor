from fastapi import Depends, Query, Request

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.list_project_audio_links.dtos.list_project_audio_links_dto_in import ListProjectAudioLinksDtoIn


def list_project_audio_links(
    request: Request,
    project_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    actor: AuthorizedActor = Depends(RequirePermission("project_audio.read")),
):
    result = request.app.state.container.list_project_audio_links.exec(
        ListProjectAudioLinksDtoIn(actor, project_id, limit, offset)
    )
    return {"success": True, "data": result.data}
