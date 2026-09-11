from fastapi import Depends, Request

from app.http.dependencies.require_permission import RequirePermission
from app.http.schemas.project_audio_input import ProjectAudioInput
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.create_project_audio_link.dtos.create_project_audio_link_dto_in import CreateProjectAudioLinkDtoIn


def create_project_audio_link(
    request: Request,
    project_id: str,
    body: ProjectAudioInput,
    actor: AuthorizedActor = Depends(RequirePermission("project_audio.update")),
):
    result = request.app.state.container.create_project_audio_link.exec(
        CreateProjectAudioLinkDtoIn(actor, project_id, body.output_id)
    )
    return {"success": True, "data": result.data}
