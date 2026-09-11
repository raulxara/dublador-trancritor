from fastapi import Depends, Request
from fastapi.responses import FileResponse

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.download_dubbing_output.dtos.download_dubbing_output_dto_in import DownloadDubbingOutputDtoIn


def download_dubbing_output(
    job_id: str,
    output_id: str,
    request: Request,
    actor: AuthorizedActor = Depends(RequirePermission("dubbing.download")),
):
    result = request.app.state.container.download_dubbing_output.exec(
        DownloadDubbingOutputDtoIn(actor, job_id, output_id)
    )
    return FileResponse(
        result.path,
        media_type="audio/wav" if result.format == "wav" else "text/plain",
        filename="result." + result.format,
        headers={"Cache-Control": "no-store"},
    )
