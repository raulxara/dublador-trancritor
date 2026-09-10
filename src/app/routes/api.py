from fastapi import APIRouter, Depends

from app.http.controllers.auth_controller import get_context, issue_token, revoke_token
from app.http.controllers.health_controller import get_health, get_readiness
from app.http.dependencies.resolve_actor import resolve_actor

router = APIRouter()
router.add_api_route("/health", get_health, methods=["GET"])
router.add_api_route("/ready", get_readiness, methods=["GET"])
# No business route is mounted until authentication and persistence are implemented.


protected = APIRouter(dependencies=[Depends(resolve_actor)])
protected.add_api_route("/auth/context", get_context, methods=["GET"])
protected.add_api_route("/users/{user_id}/access-token", issue_token, methods=["POST"], status_code=201)
protected.add_api_route("/auth/access-token", revoke_token, methods=["DELETE"])
router.include_router(protected)
