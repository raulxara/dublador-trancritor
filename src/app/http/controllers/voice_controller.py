from dataclasses import asdict

from fastapi import Depends, Query, Request

from app.http.dependencies.require_permission import RequirePermission
from app.http.schemas.voice_input import VoiceInput, VoiceUpdateInput
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.get_voice.dtos.get_voice_dto_in import GetVoiceDtoIn
from app.use_cases.list_voice_catalog.dtos.list_voice_catalog_dto_in import ListVoiceCatalogDtoIn
from app.use_cases.list_voices.dtos.list_voices_dto_in import ListVoicesDtoIn
from app.use_cases.manage_voice.dtos.manage_voice_dto_in import ManageVoiceDtoIn


def create_voice(
    body: VoiceInput, request: Request, actor: AuthorizedActor = Depends(RequirePermission("voice.register"))
):
    result = request.app.state.container.manage_voice.exec(ManageVoiceDtoIn(actor, **body.model_dump()))
    return {"success": True, "data": asdict(result)}


def update_voice(
    voice_id: str,
    body: VoiceUpdateInput,
    request: Request,
    actor: AuthorizedActor = Depends(RequirePermission("voice.update")),
):
    result = request.app.state.container.manage_voice.exec(
        ManageVoiceDtoIn(actor, unique_id=voice_id, **body.model_dump())
    )
    return {"success": True, "data": asdict(result)}


def get_voice(voice_id: str, request: Request, actor: AuthorizedActor = Depends(RequirePermission("voice.read"))):
    result = request.app.state.container.get_voice.exec(GetVoiceDtoIn(actor, voice_id))
    return {"success": True, "data": result.data}


def list_voices(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    actor: AuthorizedActor = Depends(RequirePermission("voice.read")),
):
    result = request.app.state.container.list_voices.exec(ListVoicesDtoIn(actor, limit, offset))
    return {"success": True, "data": result.data, "limit": limit, "offset": offset}


def languages(request: Request, actor: AuthorizedActor = Depends(RequirePermission("catalog.read"))):
    return {
        "success": True,
        "data": request.app.state.container.list_voice_catalog.exec(ListVoiceCatalogDtoIn(actor, "languages")).data,
    }


def gender(request: Request, actor: AuthorizedActor = Depends(RequirePermission("catalog.read"))):
    return {
        "success": True,
        "data": request.app.state.container.list_voice_catalog.exec(ListVoiceCatalogDtoIn(actor, "gender")).data,
    }
