from fastapi import APIRouter, Depends

from app.http.controllers.auth_controller import get_context, issue_token, revoke_token
from app.http.controllers.health_controller import get_health, get_readiness
from app.http.controllers.voice_controller import create_voice, gender, get_voice, languages, list_voices, update_voice
from app.http.controllers.voice_sample_controller import download_sample, list_samples, register_sample
from app.http.dependencies.resolve_actor import resolve_actor
from app.routes.catalog import router as catalog_router
from app.routes.chat import router as chat_router

router = APIRouter()
router.add_api_route("/health", get_health, methods=["GET"])
router.add_api_route("/ready", get_readiness, methods=["GET"])


protected = APIRouter(dependencies=[Depends(resolve_actor)])
protected.add_api_route("/auth/context", get_context, methods=["GET"])
protected.add_api_route("/users/{user_id}/access-token", issue_token, methods=["POST"], status_code=201)
protected.add_api_route("/auth/access-token", revoke_token, methods=["DELETE"])
protected.add_api_route("/voices", create_voice, methods=["POST"], status_code=201)
protected.add_api_route("/voices", list_voices, methods=["GET"])
protected.add_api_route("/voices/{voice_id}", get_voice, methods=["GET"])
protected.add_api_route("/voices/{voice_id}", update_voice, methods=["PUT"])
protected.add_api_route("/languages", languages, methods=["GET"])
protected.add_api_route("/gender", gender, methods=["GET"])
protected.add_api_route("/voices/{voice_id}/samples", register_sample, methods=["POST"], status_code=201)
protected.add_api_route("/voices/{voice_id}/samples", list_samples, methods=["GET"])
protected.add_api_route("/voices/{voice_id}/samples/{sample_id}/audio", download_sample, methods=["GET"])
protected.include_router(chat_router)
protected.include_router(catalog_router)
router.include_router(protected)
