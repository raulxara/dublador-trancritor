from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.config.settings import Settings
from app.entities.dubbing_job.execution_result_entity import ExecutionResultEntity
from app.entities.dubbing_job.segment_entity import SegmentEntity
from app.models.processing.sqlalchemy_execution_results_repository import SqlAlchemyExecutionResultsRepository
from app.models.processing.sqlalchemy_job_execution_repository import SqlAlchemyJobExecutionRepository
from app.services.processing.private_result_storage import PrivateResultStorage
from tests.voice_scenarios import wav_content


def check_processing(client, headers, container, office_id):
    queue = SqlAlchemyJobExecutionRepository(container.engine)
    results = SqlAlchemyExecutionResultsRepository(container.engine)
    chat = client.post("/api/v1/chats", headers=headers, json={"title": "Processing"}).json()["data"]["unique_id"]
    root = "/api/v1/chats/" + chat
    message = client.post(
        root + "/audio", headers=dict(headers, **{"Content-Type": "audio/wav"}), content=wav_content()
    ).json()["data"]["unique_id"]

    def submit(key):
        response = client.post(
            root + "/jobs",
            headers=dict(headers, **{"Idempotency-Key": key}),
            json=dict(operation="transcribe", input_message_id=message),
        )
        assert response.status_code == 202, response.text
        return response.json()["data"]["unique_id"]

    ids = {submit("processing-1"), submit("processing-2")}
    with ThreadPoolExecutor(max_workers=2) as pool:
        leases = list(pool.map(lambda n: queue.claim(str(n) * 64, 120, 3), [1, 2]))
    assert {lease.job_id for lease in leases} == ids
    assert queue.claim("3" * 64, 120, 3) is None
    lease = leases[0]
    assert queue.renew(lease, 120)
    assert not queue.renew(replace(lease, token="4" * 64), 120)
    with container.engine.begin() as connection:
        connection.execute(
            text("UPDATE dubbing_jobs SET locked_until=UTC_TIMESTAMP()-INTERVAL 1 SECOND WHERE _id=:id"),
            dict(id=lease.job_id),
        )
    assert not queue.renew(lease, 120)
    assert queue.recover(3) == 1
    replacement = queue.claim("5" * 64, 120, 3)
    assert replacement.job_id == lease.job_id and replacement.attempt == 2
    assert not queue.fail(lease, "ENGINE_FAILED")
    context = results.resolve(replacement)
    assert context and context.operation == "transcribe"
    with TemporaryDirectory() as directory:
        Path(directory, "transcript.txt").write_text("Texto de teste", encoding="utf8")
        file = PrivateResultStorage(Settings().media_root).save(office_id, directory, "transcript.txt")
    result = ExecutionResultEntity(
        (file,), "Texto de teste", "pt", (SegmentEntity(0, 1000, "Texto de teste"),), "test-only", "fixture"
    )
    assert not results.complete(lease, result)
    # A failure after inserting the assistant message must roll back the entire publication.
    with pytest.raises(IntegrityError):
        results.complete(
            replacement,
            replace(result, files=(replace(file, unique_id="foreign-file"), replace(file, unique_id="foreign-file"))),
        )
    assert results.complete(replacement, result)
    assert not results.complete(replacement, result)
    path = "/api/v1/jobs/" + replacement.job_id
    outputs = client.get(path + "/outputs", headers=headers)
    assert outputs.status_code == 200, outputs.text
    assert len(outputs.json()["data"]) == 1 and "storage_key" not in outputs.text
    output = outputs.json()["data"][0]["unique_id"]
    download = path + "/outputs/" + output + "/file"
    response = client.get(download, headers=headers)
    assert response.status_code == 200 and response.text == "Texto de teste"
    assert response.headers["cache-control"] == "no-store"
    assert client.get(download).status_code == 401
    for office, user in [(office_id, "peer-user"), ("foreign-office", "foreign-user")]:
        token, _ = container.access_token.issue(office, user)
        other = {"Authorization": "Bearer " + token}
        assert client.get(path + "/outputs", headers=other).status_code == 404
        assert client.get(download, headers=other).status_code == 404
    assert client.post(path + "/cancel", headers=headers).status_code == 409
    with container.engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT COUNT(*) FROM transcriptions WHERE job_id=:id"), dict(id=replacement.job_id)
            ).scalar_one()
            == 1
        )
        assert (
            connection.execute(
                text("SELECT COUNT(*) FROM dubbing_segments WHERE job_id=:id"), dict(id=replacement.job_id)
            ).scalar_one()
            == 1
        )
    running = leases[1]
    assert client.post("/api/v1/jobs/" + running.job_id + "/cancel", headers=headers).status_code == 200
    assert not queue.renew(running, 120) and not results.complete(running, result)
    submit("retry-limit")
    for attempt in range(1, 4):
        retry = queue.claim(str(attempt) * 64, 120, 3)
        assert retry.attempt == attempt
        with container.engine.begin() as connection:
            connection.execute(
                text("UPDATE dubbing_jobs SET locked_until=UTC_TIMESTAMP()-INTERVAL 1 SECOND WHERE _id=:id"),
                dict(id=retry.job_id),
            )
        assert queue.recover(3) == 1
    assert queue.claim("6" * 64, 120, 3) is None
    assert client.get("/api/v1/jobs/" + retry.job_id, headers=headers).json()["data"]["processing_state"] == "failed"
