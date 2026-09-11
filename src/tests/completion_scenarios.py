import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from sqlalchemy import text

from app.config.settings import Settings
from app.entities.dubbing_job.execution_result_entity import ExecutionResultEntity
from app.entities.dubbing_job.result_file_entity import ResultFileEntity
from app.models.processing.sqlalchemy_execution_results_repository import SqlAlchemyExecutionResultsRepository
from app.models.processing.sqlalchemy_job_execution_repository import SqlAlchemyJobExecutionRepository
from app.services.voice.private_media_storage import PrivateMediaStorage
from tests.voice_scenarios import wav_content


def check_completion(client, headers, container, office_id):
    with container.engine.connect() as connection:
        voice = connection.execute(
            text(
                "SELECT _id FROM voices WHERE office_id=:office AND current_sample_id IS NOT NULL "
                'AND status="active" LIMIT 1'
            ),
            dict(office=office_id),
        ).scalar_one()
        transcription = (
            connection.execute(
                text("SELECT _id,job_id FROM transcriptions WHERE office_id=:office LIMIT 1"), dict(office=office_id)
            )
            .mappings()
            .one()
        )
    assert client.get("/api/v1/tags").status_code == 401
    payload = dict(name="Narrador", slug="narrador")
    response = client.post("/api/v1/tags", headers=headers, json=payload)
    assert response.status_code == 201, response.text
    tag = response.json()["data"]["unique_id"]
    assert client.post("/api/v1/tags", headers=headers, json=payload).status_code == 409
    assert (
        client.post("/api/v1/tags", headers=headers, json=dict(payload, office_id="foreign-office")).status_code == 422
    )
    path = "/api/v1/voices/" + voice + "/tags"
    assert client.put(path, headers=headers, json={"tag_ids": [tag, tag]}).status_code == 422
    assert client.put(path, headers=headers, json={"tag_ids": [tag]}).status_code == 200
    assert client.get(path, headers=headers).json()["data"][0]["unique_id"] == tag
    assert client.put(path, headers=headers, json={"tag_ids": ["foreign-tag"]}).status_code == 404
    assert len(client.get(path, headers=headers).json()["data"]) == 1
    assert client.put("/api/v1/tags/" + tag, headers=headers, json=dict(payload, status="inactive")).status_code == 200
    assert client.get(path, headers=headers).json()["data"] == []
    assert client.put(path, headers=headers, json={"tag_ids": [tag]}).status_code == 404
    assert client.put("/api/v1/tags/" + tag, headers=headers, json=dict(payload, status="active")).status_code == 200
    assert client.put(path, headers=headers, json={"tag_ids": []}).status_code == 200
    trans_path = "/api/v1/jobs/" + transcription["job_id"] + "/transcriptions"
    body = dict(base_transcription_id=transcription["_id"], text="Texto revisado")
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: client.post(trans_path, headers=headers, json=body), range(2)))
    assert sorted(r.status_code for r in responses) == [201, 409], [r.text for r in responses]
    versions = client.get(trans_path, headers=headers).json()["data"]
    assert [v["version"] for v in versions] == [2, 1]
    assert versions[0]["origin"] == "edited" and versions[0]["previous_transcription_id"] == transcription["_id"]
    assert versions[1]["text"] == "Texto de teste"
    with container.engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT recognized_text FROM dubbing_segments WHERE job_id=:id"), dict(id=transcription["job_id"])
            ).scalar_one()
            == "Texto de teste"
        )
    chat = client.post("/api/v1/chats", headers=headers, json=dict(title="Projects", selected_voice_id=voice)).json()[
        "data"
    ]["unique_id"]
    with TemporaryDirectory() as directory:
        source = Path(directory) / "source.wav"
        source.write_bytes(wav_content())
        for extension, mime, codec in [("mp3", "audio/mpeg", "libmp3lame"), ("mp4", "video/mp4", "aac")]:
            encoded = Path(directory) / ("input." + extension)
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-c:a", codec, str(encoded)], check=True
            )
            response = client.post(
                "/api/v1/chats/" + chat + "/audio",
                headers=dict(headers, **{"Content-Type": mime}),
                content=encoded.read_bytes(),
            )
            assert response.status_code == 201, response.text
            message = response.json()["data"]["unique_id"]
            audio = client.get("/api/v1/chats/" + chat + "/messages/" + message + "/audio", headers=headers)
            assert audio.status_code == 200 and audio.content.startswith(b"RIFF")
    job = client.post(
        "/api/v1/chats/" + chat + "/jobs",
        headers=dict(headers, **{"Idempotency-Key": "project-output"}),
        json=dict(operation="text_to_speech", input_text="Example"),
    ).json()["data"]["unique_id"]
    lease = SqlAlchemyJobExecutionRepository(container.engine).claim("9" * 64, 120, 3)
    assert lease.job_id == job
    identifier = str(uuid4())
    content = wav_content()
    key = PrivateMediaStorage(Settings().media_root).save(office_id, identifier, content)
    file = ResultFileEntity(identifier, key, "wav", "audio", len(content), "a" * 64, 3000, 16000, 1)
    assert SqlAlchemyExecutionResultsRepository(container.engine).complete(
        lease, ExecutionResultEntity((file,), None, "pt", (), "test", "fixture")
    )
    output = client.get("/api/v1/jobs/" + job + "/outputs", headers=headers).json()["data"][0]["unique_id"]
    project = "/api/v1/projects/external-project-1/audio-links"
    response = client.post(project, headers=headers, json={"output_id": output})
    assert response.status_code == 201, response.text
    link = response.json()["data"]["unique_id"]
    assert client.post(project, headers=headers, json={"output_id": output}).json()["data"]["unique_id"] == link
    assert len(client.get(project, headers=headers).json()["data"]) == 1
    assert client.delete("/api/v1/project-audio-links/" + link, headers=headers).status_code == 200
    assert client.get(project, headers=headers).json()["data"] == []
    assert client.post(project, headers=headers, json={"output_id": output}).json()["data"]["unique_id"] == link
    with container.engine.begin() as connection:
        connection.execute(
            text("""INSERT IGNORE INTO position_permission (_id,office_id,position_id,permission_id)
            SELECT UUID(),p.office_id,p._id,perm._id FROM positions p CROSS JOIN permissions perm
            WHERE perm.office_id IS NULL AND perm.entity IN ('tag','transcription','project_audio')""")
        )
    for office, user in [(office_id, "peer-user"), ("foreign-office", "foreign-user")]:
        token, _ = container.access_token.issue(office, user)
        other = {"Authorization": "Bearer " + token}
        assert client.get(trans_path, headers=other).status_code == 404
        assert client.post(trans_path, headers=other, json=body).status_code == 404
        assert client.post(project, headers=other, json={"output_id": output}).status_code == 404
        assert client.get(project, headers=other).json()["data"] == []
        assert client.delete("/api/v1/project-audio-links/" + link, headers=other).status_code == 404
        if office != office_id:
            assert (
                client.put("/api/v1/tags/" + tag, headers=other, json=dict(payload, status="active")).status_code == 404
            )
    with container.engine.begin() as connection:
        connection.execute(
            text("UPDATE permissions SET status='inactive' WHERE entity IN ('tag','project_audio','transcription')")
        )
    assert client.get("/api/v1/tags", headers=headers).status_code == 403
    assert client.post(trans_path, headers=headers, json=body).status_code == 403
    assert client.get(project, headers=headers).status_code == 403
    with container.engine.begin() as connection:
        connection.execute(
            text("UPDATE permissions SET status='active' WHERE entity IN ('tag','project_audio','transcription')")
        )
