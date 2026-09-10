from fastapi import Depends, Request, Response

from app.http.dependencies.require_permission import RequirePermission
from app.http.dependencies.resolve_actor import resolve_actor
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.issue_access_token.dtos.issue_access_token_dto_in import IssueAccessTokenDtoIn


def get_context(response: Response, actor: AuthorizedActor = Depends(RequirePermission("catalog.read"))):
    response.headers["Cache-Control"] = "no-store"
    return {
        "success": True,
        "data": {
            "id": actor.user_customer_id,
            "officeId": actor.office_id,
            "userId": actor.user_id,
            "profileId": actor.profile_id,
            "permissions": sorted(actor.permissions),
        },
    }


def issue_token(
    user_id: str,
    request: Request,
    response: Response,
    actor: AuthorizedActor = Depends(RequirePermission("user.update")),
):
    result = request.app.state.container.issue_access_token.exec(IssueAccessTokenDtoIn(actor, user_id))
    response.headers["Cache-Control"] = "no-store"
    return {"success": True, "data": {"userId": result.user_id, "token": result.token, "expiresAt": result.expires_at}}


def revoke_token(request: Request, actor: AuthorizedActor = Depends(resolve_actor)):
    request.app.state.container.access_token.revoke(actor.office_id, actor.user_customer_id)
    return Response(status_code=204, headers={"Cache-Control": "no-store"})
