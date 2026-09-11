from fastapi import Depends, Request

from app.http.dependencies.require_permission import RequirePermission
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.list_dubbing_outputs.dtos.list_dubbing_outputs_dto_in import ListDubbingOutputsDtoIn


def list_dubbing_outputs(
    job_id: str, request: Request, actor: AuthorizedActor = Depends(RequirePermission("dubbing.read"))
):
    result = request.app.state.container.list_dubbing_outputs.exec(ListDubbingOutputsDtoIn(actor, job_id))
    return {"success": True, "data": result.data}
