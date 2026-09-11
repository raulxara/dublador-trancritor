from fastapi import Depends, Request

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.cancel_dubbing_job.dtos.cancel_dubbing_job_dto_in import CancelDubbingJobDtoIn


def cancel_dubbing_job(
    job_id: str, request: Request, actor: AuthorizedActor = Depends(RequirePermission("dubbing.cancel"))
):
    dto = CancelDubbingJobDtoIn(actor, job_id)
    result = request.app.state.container.cancel_dubbing_job.exec(dto)
    return {"success": True, "data": result.data}
