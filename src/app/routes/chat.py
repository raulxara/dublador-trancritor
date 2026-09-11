from fastapi import APIRouter

from app.http.controllers.chat.cancel_dubbing_job_controller import cancel_dubbing_job
from app.http.controllers.chat.create_dubbing_chat_controller import create_dubbing_chat
from app.http.controllers.chat.download_chat_audio_controller import download_chat_audio
from app.http.controllers.chat.download_dubbing_output_controller import download_dubbing_output
from app.http.controllers.chat.get_dubbing_chat_controller import get_dubbing_chat
from app.http.controllers.chat.get_dubbing_job_controller import get_dubbing_job
from app.http.controllers.chat.list_dubbing_chats_controller import list_dubbing_chats
from app.http.controllers.chat.list_dubbing_jobs_controller import list_dubbing_jobs
from app.http.controllers.chat.list_dubbing_messages_controller import list_dubbing_messages
from app.http.controllers.chat.list_dubbing_outputs_controller import list_dubbing_outputs
from app.http.controllers.chat.submit_dubbing_job_controller import submit_dubbing_job
from app.http.controllers.chat.update_dubbing_chat_controller import update_dubbing_chat
from app.http.controllers.chat.upload_chat_audio_controller import upload_chat_audio

router = APIRouter()
router.add_api_route("/chats", create_dubbing_chat, methods=["POST"], status_code=201)
router.add_api_route("/chats/{chat_id}", update_dubbing_chat, methods=["PUT"], status_code=200)
router.add_api_route("/chats/{chat_id}", get_dubbing_chat, methods=["GET"], status_code=200)
router.add_api_route("/chats", list_dubbing_chats, methods=["GET"], status_code=200)
router.add_api_route("/chats/{chat_id}/messages", list_dubbing_messages, methods=["GET"], status_code=200)
router.add_api_route("/chats/{chat_id}/jobs", submit_dubbing_job, methods=["POST"], status_code=202)
router.add_api_route("/jobs/{job_id}", get_dubbing_job, methods=["GET"], status_code=200)
router.add_api_route("/jobs/{job_id}/cancel", cancel_dubbing_job, methods=["POST"], status_code=200)
router.add_api_route("/chats/{chat_id}/audio", upload_chat_audio, methods=["POST"], status_code=201)
router.add_api_route(
    "/chats/{chat_id}/messages/{message_id}/audio", download_chat_audio, methods=["GET"], status_code=200
)

router.add_api_route("/chats/{chat_id}/jobs", list_dubbing_jobs, methods=["GET"])

router.add_api_route("/jobs/{job_id}/outputs", list_dubbing_outputs, methods=["GET"])

router.add_api_route("/jobs/{job_id}/outputs/{output_id}/file", download_dubbing_output, methods=["GET"])
