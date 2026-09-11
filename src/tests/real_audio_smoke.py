"""Manual Docker-only smoke; requires an empty disposable DB and local model/sample mounts."""

import io
import os
import wave
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config.settings import Settings
from app.config.worker_settings import WorkerSettings
from app.main import create_app
from app.providers.container import Container
from app.providers.worker_container import WorkerContainer
from app.use_cases.bootstrap_access.dtos.bootstrap_access_dto_in import BootstrapAccessDtoIn
from app.use_cases.process_next_job.dtos.process_next_job_dto_in import ProcessNextJobDtoIn


def main():
    settings = Settings()
    if os.environ.get("DUBBER_INTEGRATION") != "1" or settings.db_database != "auth_test":
        raise SystemExit("Only an isolated auth_test database is allowed.")
    migration = Config("alembic.ini")
    command.upgrade(migration, "0003_chat_jobs")
    container = Container.build(settings)
    result = container.bootstrap_access.exec(
        BootstrapAccessDtoIn("Smoke", "smoke", "Test", "User", "smoke@example.test", "smoke", "test-password-long")
    )
    with container.engine.connect() as connection:
        token_before = connection.execute(text("SELECT token FROM user_customers")).scalar_one()
    command.upgrade(migration, "head")
    with container.engine.connect() as connection:
        assert connection.execute(text("SELECT token FROM user_customers")).scalar_one() == token_before
    print("upgrade_preserved_credential", flush=True)
    headers = {"Authorization": "Bearer " + result.token}
    worker = WorkerContainer.build(settings, WorkerSettings())
    worker.audio.preflight()
    try:
        with TestClient(create_app(settings)) as client:
            language = next(
                item["_id"]
                for item in client.get("/api/v1/languages", headers=headers).json()["data"]
                if item["slug"].lower().startswith("pt")
            )
            voice_response = client.post(
                "/api/v1/voices", headers=headers, json=dict(name="Smoke voice", language_id=language)
            )
            assert voice_response.status_code == 201, voice_response.text
            voice = voice_response.json()["data"]["unique_id"]
            audio_headers = dict(headers, **{"Content-Type": "audio/wav"})
            sample = Path("/sample/clean.wav").read_bytes()
            uploaded = client.post("/api/v1/voices/" + voice + "/samples", headers=audio_headers, content=sample)
            assert uploaded.status_code == 201, uploaded.text
            chat = client.post(
                "/api/v1/chats", headers=headers, json=dict(title="Smoke", selected_voice_id=voice)
            ).json()["data"]["unique_id"]

            def execute(key, body):
                response = client.post(
                    "/api/v1/chats/" + chat + "/jobs", headers=dict(headers, **{"Idempotency-Key": key}), json=body
                )
                assert response.status_code == 202, response.text
                job = response.json()["data"]["unique_id"]
                outcome = worker.process.exec(ProcessNextJobDtoIn())
                assert outcome.state == "completed", outcome.state
                outputs = client.get("/api/v1/jobs/" + job + "/outputs", headers=headers).json()["data"]
                artifacts = {}
                for output in outputs:
                    downloaded = client.get(
                        "/api/v1/jobs/" + job + "/outputs/" + output["unique_id"] + "/file", headers=headers
                    )
                    assert downloaded.status_code == 200
                    artifacts[output["format"]] = downloaded.content
                print(key + "_completed", flush=True)
                return artifacts

            generated = execute(
                "tts",
                dict(
                    operation="text_to_speech",
                    input_text=(
                        "Olá, este é um teste do serviço de dublagem. A aplicação transforma este texto em áudio."
                    ),
                ),
            )
            with wave.open(io.BytesIO(generated["wav"])) as audio:
                frames = audio.readframes(audio.getnframes())
                rate = audio.getframerate()
            # Respect the API's existing three-second upload minimum, independently of model speaking speed.
            frames += b"\0\0" * max(0, rate * 3 - len(frames) // 2)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(rate)
                audio.writeframes(frames)
            source = client.post("/api/v1/chats/" + chat + "/audio", headers=audio_headers, content=buffer.getvalue())
            assert source.status_code == 201, source.text
            message = source.json()["data"]["unique_id"]
            transcript = execute("asr", dict(operation="transcribe", input_message_id=message))
            assert transcript["txt"].strip()
            converted = execute(
                "conversion", dict(operation="speech_to_speech", input_message_id=message, preserve_timing=True)
            )
            with wave.open(io.BytesIO(converted["wav"])) as audio:
                assert abs(audio.getnframes() / audio.getframerate() - len(frames) / 2 / rate) < 0.01
            assert converted["txt"].strip()
            print("real_audio_all_operations_passed", flush=True)
    finally:
        worker.engine.dispose()
        container.close()


if __name__ == "__main__":
    main()
