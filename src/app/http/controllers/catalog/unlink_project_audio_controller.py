from fastapi import Depends, Request

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.unlink_project_audio.dtos.unlink_project_audio_dto_in import (
    UnlinkProjectAudioDtoIn,
)


def unlink_project_audio(
    request: Request, link_id: str, actor: AuthorizedActor = Depends(RequirePermission("project_audio.update"))
):
    result = request.app.state.container.unlink_project_audio.exec(UnlinkProjectAudioDtoIn(actor, link_id))
    return {"success": True, "data": result.data}
