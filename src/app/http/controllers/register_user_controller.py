from fastapi import Depends, Request, Response

from app.http.dependencies.require_permission import RequirePermission
from app.http.requests.register_user_request import RegisterUserRequest
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.register_user.dtos.register_user_dto_in import RegisterUserDtoIn


def register_user(
    body: RegisterUserRequest,
    request: Request,
    response: Response,
    actor: AuthorizedActor = Depends(RequirePermission("user.register")),
):
    result = request.app.state.container.register_user.exec(
        RegisterUserDtoIn(
            actor=actor,
            first_name=body.firstName,
            last_name=body.lastName,
            email=body.email,
            phone=body.phone,
            username=body.username,
            password=body.password.get_secret_value(),
            document_type=body.documentType,
            document_value=body.documentValue,
            office_id=body.officeId,
        )
    )
    response.headers["Cache-Control"] = "no-store"
    return {
        "success": True,
        "data": {
            "officeId": result.office_id,
            "profileId": result.profile_id,
            "userId": result.user_id,
            "userCustomerId": result.user_customer_id,
            "userPositionId": result.user_position_id,
            "positionId": result.position_id,
            "assignedPositionSlug": "customer",
            "userType": "customer",
            "status": "active",
            "token": result.token,
            "expiresAt": result.expires_at,
        },
    }
