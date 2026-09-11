from fastapi import Depends, Request

from app.http.dependencies.require_permission import RequirePermission
from app.http.schemas.voice_tags_input import VoiceTagsInput
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.set_voice_tags.dtos.set_voice_tags_dto_in import SetVoiceTagsDtoIn


def set_voice_tags(
    request: Request,
    voice_id: str,
    body: VoiceTagsInput,
    actor: AuthorizedActor = Depends(RequirePermission("voice.update")),
):
    result = request.app.state.container.set_voice_tags.exec(SetVoiceTagsDtoIn(actor, voice_id, tuple(body.tag_ids)))
    return {"success": True, "data": result.data}
