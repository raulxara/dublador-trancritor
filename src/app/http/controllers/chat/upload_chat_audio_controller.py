from fastapi import Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.services.voice.validate_wav_service import ValidateWavService
from app.use_cases.upload_chat_audio.dtos.upload_chat_audio_dto_in import UploadChatAudioDtoIn


async def upload_chat_audio(
    chat_id: str, request: Request, actor: AuthorizedActor = Depends(RequirePermission("dubbing.generate"))
):
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in ("audio/wav", "audio/x-wav", "audio/mpeg", "video/mp4"):
        raise HTTPException(status_code=415, detail="Use audio/wav, audio/mpeg or video/mp4")
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > ValidateWavService.MAX_BYTES:
            raise HTTPException(status_code=413, detail="Maximum size: 20 MiB")
        content.extend(chunk)
    result = await run_in_threadpool(
        request.app.state.container.upload_chat_audio.exec,
        UploadChatAudioDtoIn(actor, chat_id, bytes(content), content_type),
    )
    return {"success": True, "data": result.data}
