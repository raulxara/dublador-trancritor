from fastapi import Depends, Request

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.list_voice_tags.dtos.list_voice_tags_dto_in import ListVoiceTagsDtoIn


def list_voice_tags(request: Request, voice_id: str, actor: AuthorizedActor = Depends(RequirePermission("voice.read"))):
    result = request.app.state.container.list_voice_tags.exec(ListVoiceTagsDtoIn(actor, voice_id))
    return {"success": True, "data": result.data}
