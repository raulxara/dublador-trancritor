from dataclasses import asdict

from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.services.voice.validate_wav_service import ValidateWavService
from app.use_cases.download_voice_sample.dtos.download_voice_sample_dto_in import DownloadVoiceSampleDtoIn
from app.use_cases.list_voice_samples.dtos.list_voice_samples_dto_in import ListVoiceSamplesDtoIn
from app.use_cases.register_voice_sample.dtos.register_voice_sample_dto_in import RegisterVoiceSampleDtoIn


async def register_sample(
    voice_id: str, request: Request, actor: AuthorizedActor = Depends(RequirePermission("voice.update"))
):
    if request.headers.get("content-type", "").split(";")[0].lower() not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(status_code=415, detail="Use audio/wav")
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > ValidateWavService.MAX_BYTES:
            raise HTTPException(status_code=413, detail="Maximum size: 20 MiB")
        content.extend(chunk)
    result = await run_in_threadpool(
        request.app.state.container.register_voice_sample.exec,
        RegisterVoiceSampleDtoIn(actor, voice_id, bytes(content)),
    )
    return {"success": True, "data": asdict(result)}


def list_samples(
    voice_id: str,
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    actor: AuthorizedActor = Depends(RequirePermission("voice.read")),
):
    data = request.app.state.container.list_voice_samples.exec(
        ListVoiceSamplesDtoIn(actor, voice_id, limit, offset)
    ).data
    return {"success": True, "data": data, "limit": limit, "offset": offset}


def download_sample(
    voice_id: str, sample_id: str, request: Request, actor: AuthorizedActor = Depends(RequirePermission("voice.read"))
):
    path = request.app.state.container.download_voice_sample.exec(
        DownloadVoiceSampleDtoIn(actor, voice_id, sample_id)
    ).data
    return FileResponse(path, media_type="audio/wav", filename="sample.wav", headers={"Cache-Control": "no-store"})
