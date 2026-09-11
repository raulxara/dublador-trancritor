from fastapi import Depends, Request

from app.http.dependencies.require_permission import RequirePermission
from app.http.schemas.tag_input import TagInput
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.create_tag.dtos.create_tag_dto_in import CreateTagDtoIn


def create_tag(request: Request, body: TagInput, actor: AuthorizedActor = Depends(RequirePermission("tag.update"))):
    result = request.app.state.container.create_tag.exec(CreateTagDtoIn(actor, body.name, body.slug))
    return {"success": True, "data": result.data}
