from fastapi import Depends, Request

from app.http.dependencies.require_permission import RequirePermission
from app.http.schemas.edit_transcription_input import EditTranscriptionInput
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.edit_transcription.dtos.edit_transcription_dto_in import EditTranscriptionDtoIn


def edit_transcription(
    request: Request,
    job_id: str,
    body: EditTranscriptionInput,
    actor: AuthorizedActor = Depends(RequirePermission("transcription.update")),
):
    result = request.app.state.container.edit_transcription.exec(
        EditTranscriptionDtoIn(actor, job_id, body.base_transcription_id, body.text)
    )
    return {"success": True, "data": result.data}
