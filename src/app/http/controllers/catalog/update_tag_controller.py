from fastapi import Depends, Request

from app.http.dependencies.require_permission import RequirePermission
from app.http.schemas.update_tag_input import UpdateTagInput
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.update_tag.dtos.update_tag_dto_in import UpdateTagDtoIn


def update_tag(
    request: Request,
    tag_id: str,
    body: UpdateTagInput,
    actor: AuthorizedActor = Depends(RequirePermission("tag.update")),
):
    result = request.app.state.container.update_tag.exec(
        UpdateTagDtoIn(actor, tag_id, body.name, body.slug, body.status)
    )
    return {"success": True, "data": result.data}
