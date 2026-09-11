from fastapi import APIRouter

from app.http.controllers.catalog.create_project_audio_link_controller import create_project_audio_link
from app.http.controllers.catalog.create_tag_controller import create_tag
from app.http.controllers.catalog.edit_transcription_controller import edit_transcription
from app.http.controllers.catalog.list_project_audio_links_controller import list_project_audio_links
from app.http.controllers.catalog.list_tags_controller import list_tags
from app.http.controllers.catalog.list_transcriptions_controller import list_transcriptions
from app.http.controllers.catalog.list_voice_tags_controller import list_voice_tags
from app.http.controllers.catalog.set_voice_tags_controller import set_voice_tags
from app.http.controllers.catalog.unlink_project_audio_controller import unlink_project_audio
from app.http.controllers.catalog.update_tag_controller import update_tag

router = APIRouter()
router.add_api_route("/tags", create_tag, methods=["POST"], status_code=201)
router.add_api_route("/tags/{tag_id}", update_tag, methods=["PUT"], status_code=200)
router.add_api_route("/tags", list_tags, methods=["GET"], status_code=200)
router.add_api_route("/voices/{voice_id}/tags", list_voice_tags, methods=["GET"], status_code=200)
router.add_api_route("/voices/{voice_id}/tags", set_voice_tags, methods=["PUT"], status_code=200)
router.add_api_route("/projects/{project_id}/audio-links", create_project_audio_link, methods=["POST"], status_code=201)
router.add_api_route("/projects/{project_id}/audio-links", list_project_audio_links, methods=["GET"], status_code=200)
router.add_api_route("/project-audio-links/{link_id}", unlink_project_audio, methods=["DELETE"], status_code=200)
router.add_api_route("/jobs/{job_id}/transcriptions", list_transcriptions, methods=["GET"], status_code=200)
router.add_api_route("/jobs/{job_id}/transcriptions", edit_transcription, methods=["POST"], status_code=201)
