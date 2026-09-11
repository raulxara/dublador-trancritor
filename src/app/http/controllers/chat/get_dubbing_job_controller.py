from fastapi import Depends, Request

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.get_dubbing_job.dtos.get_dubbing_job_dto_in import GetDubbingJobDtoIn


def get_dubbing_job(job_id: str, request: Request, actor: AuthorizedActor = Depends(RequirePermission("dubbing.read"))):
    dto = GetDubbingJobDtoIn(actor, job_id)
    result = request.app.state.container.get_dubbing_job.exec(dto)
    return {"success": True, "data": result.data}
