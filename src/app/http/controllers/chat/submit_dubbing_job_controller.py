from fastapi import Depends, Header, Request

from app.http.dependencies.require_permission import RequirePermission
from app.http.schemas.submit_dubbing_job_input import SubmitDubbingJobInput
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.submit_dubbing_job.dtos.submit_dubbing_job_dto_in import SubmitDubbingJobDtoIn


def submit_dubbing_job(
    chat_id: str,
    body: SubmitDubbingJobInput,
    request: Request,
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=100),
    actor: AuthorizedActor = Depends(RequirePermission("dubbing.generate")),
):
    dto = SubmitDubbingJobDtoIn(actor, chat_id, idempotency_key, **body.model_dump())
    result = request.app.state.container.submit_dubbing_job.exec(dto)
    return {"success": True, "data": result.data}
