from fastapi import Depends, Request
from fastapi.responses import FileResponse

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.download_chat_audio.dtos.download_chat_audio_dto_in import DownloadChatAudioDtoIn


def download_chat_audio(
    chat_id: str,
    message_id: str,
    request: Request,
    actor: AuthorizedActor = Depends(RequirePermission("dubbing.download")),
):
    dto = DownloadChatAudioDtoIn(actor, chat_id, message_id)
    result = request.app.state.container.download_chat_audio.exec(dto)
    return FileResponse(
        result.data, media_type="audio/wav", filename="input.wav", headers={"Cache-Control": "no-store"}
    )
