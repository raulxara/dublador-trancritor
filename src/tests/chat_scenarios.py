from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from sqlalchemy import text

from app.exceptions.invalid_input_error import InvalidInputError
from tests.voice_scenarios import wav_content


def check_chat_endpoints(client, headers, container, office_id):
    with container.engine.connect() as connection:
        voice = connection.execute(
            text("SELECT _id FROM voices WHERE office_id=:office AND current_sample_id IS NOT NULL LIMIT 1"),
            dict(office=office_id),
        ).scalar_one()
        sample = connection.execute(
            text("SELECT current_sample_id FROM voices WHERE _id=:id"), dict(id=voice)
        ).scalar_one()
    body = dict(title="My chat", selected_voice_id=voice)
    assert client.post("/api/v1/chats", json=body).status_code == 401
    assert client.post("/api/v1/chats", headers=headers, json=dict(body, user_customer_id="fake")).status_code == 422
    assert (
        client.post("/api/v1/chats", headers=headers, json=dict(body, selected_voice_id="foreign-voice")).status_code
        == 404
    )
    response = client.post("/api/v1/chats", headers=headers, json=body)
    assert response.status_code == 201, response.text
    chat = response.json()["data"]["unique_id"]
    root = "/api/v1/chats/" + chat
    assert client.get(root, headers=headers).status_code == 200
    assert len(client.get("/api/v1/chats", headers=headers).json()["data"]) == 1
    assert client.get("/api/v1/chats?limit=101", headers=headers).status_code == 422
    job_body = dict(operation="text_to_speech", input_text="Hello world")
    job_headers = dict(headers, **{"Idempotency-Key": "text-1"})
    assert client.post(root + "/jobs", headers=headers, json=job_body).status_code == 422
    assert client.post(root + "/jobs", headers=job_headers, json=dict(job_body, role="assistant")).status_code == 422
    for extra in [
        dict(input_text=" "),
        dict(input_message_id="fake"),
        dict(speed="2"),
        dict(speed="0.501"),
        dict(preserve_timing=True),
    ]:
        assert client.post(root + "/jobs", headers=job_headers, json=dict(job_body, **extra)).status_code == 422
    response = client.post(root + "/jobs", headers=job_headers, json=job_body)
    assert response.status_code == 202, response.text
    job = response.json()["data"]
    job_id = job["unique_id"]
    assert job["processing_state"] == "queued" and job["voice_sample_id"] == sample
    assert "worker_token" not in response.text and "request_hash" not in response.text
    assert (
        client.post(root + "/jobs", headers=job_headers, json=dict(job_body, speed="1.00")).json()["data"]["unique_id"]
        == job_id
    )
    assert (
        client.post(root + "/jobs", headers=job_headers, json=dict(job_body, input_text="Different")).status_code == 409
    )
    assert len(client.get(root + "/messages", headers=headers).json()["data"]) == 1
    concurrent_headers = dict(headers, **{"Idempotency-Key": "concurrent"})
    with ThreadPoolExecutor(max_workers=3) as executor:
        responses = list(
            executor.map(lambda _: client.post(root + "/jobs", headers=concurrent_headers, json=job_body), range(3))
        )
    assert all(r.status_code == 202 for r in responses), [r.text for r in responses]
    assert len({r.json()["data"]["unique_id"] for r in responses}) == 1
    assert len(client.get(root + "/messages", headers=headers).json()["data"]) == 2
    # Every mutation is part of the shared transaction.
    with patch(
        "app.models.chat.sqlalchemy_job_repository.SqlAlchemyJobRepository.create", side_effect=InvalidInputError
    ):
        assert (
            client.post(
                root + "/jobs", headers=dict(headers, **{"Idempotency-Key": "rollback"}), json=job_body
            ).status_code
            == 422
        )
    assert len(client.get(root + "/messages", headers=headers).json()["data"]) == 2
    # Changing selected voice does not mutate accepted jobs or invalidate identical retries.
    assert (
        client.put(root, headers=headers, json=dict(title="No selected voice", selected_voice_id=None)).status_code
        == 200
    )
    assert client.post(root + "/jobs", headers=job_headers, json=job_body).json()["data"]["voice_sample_id"] == sample
    assert (
        client.post(root + "/jobs", headers=dict(headers, **{"Idempotency-Key": "no-voice"}), json=job_body).status_code
        == 422
    )
    assert client.put(root, headers=headers, json=body).status_code == 200
    content = wav_content()
    audio_headers = dict(headers, **{"Content-Type": "audio/wav"})
    assert client.post(root + "/audio", headers=audio_headers, content=b"bad").status_code == 422
    upload = client.post(root + "/audio", headers=audio_headers, content=content)
    assert upload.status_code == 201, upload.text
    message = upload.json()["data"]["unique_id"]
    audio_path = root + "/messages/" + message + "/audio"
    assert client.get(audio_path, headers=headers).content == content
    speech = dict(operation="speech_to_speech", input_message_id=message, preserve_timing=True)
    converted = client.post(root + "/jobs", headers=dict(headers, **{"Idempotency-Key": "speech"}), json=speech)
    assert converted.status_code == 202, converted.text
    transcribed = client.post(
        root + "/jobs",
        headers=dict(headers, **{"Idempotency-Key": "transcribe"}),
        json=dict(operation="transcribe", input_message_id=message),
    )
    assert transcribed.status_code == 202 and transcribed.json()["data"]["voice_sample_id"] is None
    listed = client.get(root + "/jobs", headers=headers)
    assert listed.status_code == 200, listed.text
    assert len(listed.json()["data"]) == 4
    other_chat = client.post("/api/v1/chats", headers=headers, json=body).json()["data"]["unique_id"]
    assert (
        client.post(
            "/api/v1/chats/" + other_chat + "/jobs",
            headers=dict(headers, **{"Idempotency-Key": "wrong-chat"}),
            json=speech,
        ).status_code
        == 404
    )
    # Same-office user with all the same permissions still cannot access another owner's chat.
    with container.engine.begin() as connection:
        connection.execute(
            text(
                (
                    "INSERT INTO users (_id,office_id,user_type,username,password) VALUES "
                    "('peer-user',:office,'office','peer','unused')"
                )
            ),
            dict(office=office_id),
        )
        connection.execute(
            text(
                (
                    "INSERT INTO profiles (_id,office_id,first_name,last_name,email) VALUES "
                    "('peer-profile',:office,'Peer','User','peer@example.test')"
                )
            ),
            dict(office=office_id),
        )
        connection.execute(
            text(
                (
                    "INSERT INTO user_customers (_id,office_id,user_id,profile_id) VALUES "
                    "('peer-customer',:office,'peer-user','peer-profile')"
                )
            ),
            dict(office=office_id),
        )
        connection.execute(
            text(
                (
                    "INSERT INTO user_position (_id,office_id,user_id,position_id) SELECT "
                    "'peer-position',office_id,'peer-user',_id FROM positions WHERE "
                    "office_id=:office AND slug='admin'"
                )
            ),
            dict(office=office_id),
        )
    peer_token, _ = container.access_token.issue(office_id, "peer-user")
    peer_headers = {"Authorization": "Bearer " + peer_token}
    assert client.get(root, headers=peer_headers).status_code == 404
    assert client.get("/api/v1/chats", headers=peer_headers).json()["data"] == []
    assert client.get(root + "/messages", headers=peer_headers).status_code == 404
    assert client.get(root + "/jobs", headers=peer_headers).status_code == 404
    assert client.get(audio_path, headers=peer_headers).status_code == 404
    assert client.get("/api/v1/jobs/" + job_id, headers=peer_headers).status_code == 404
    assert client.post("/api/v1/jobs/" + job_id + "/cancel", headers=peer_headers).status_code == 404
    assert (
        client.post(
            root + "/jobs", headers=dict(peer_headers, **{"Idempotency-Key": "peer"}), json=job_body
        ).status_code
        == 404
    )
    assert (
        client.post(
            root + "/audio", headers=dict(peer_headers, **{"Content-Type": "audio/wav"}), content=content
        ).status_code
        == 404
    )
    assert client.put(root, headers=peer_headers, json=body).status_code == 404
    # Foreign office also has explicit permissions, so denial exercises resource scope.
    with container.engine.begin() as connection:
        connection.execute(
            text(
                (
                    "INSERT INTO positions (_id,office_id,name,slug) VALUES "
                    "('foreign-role','foreign-office','Reader','reader')"
                )
            )
        )
        connection.execute(
            text(
                (
                    "INSERT INTO user_position (_id,office_id,user_id,position_id) VALUES "
                    "('foreign-up','foreign-office','foreign-user','foreign-role')"
                )
            )
        )
        connection.execute(
            text(
                (
                    "INSERT INTO position_permission (_id,office_id,position_id,permission_id) "
                    "SELECT UUID(),'foreign-office','foreign-role',_id FROM permissions WHERE "
                    "office_id IS NULL AND entity IN ('dubbing_chat','dubbing')"
                )
            )
        )
    foreign_token, _ = container.access_token.issue("foreign-office", "foreign-user")
    foreign_headers = {"Authorization": "Bearer " + foreign_token}
    assert client.get(root, headers=foreign_headers).status_code == 404
    assert client.get("/api/v1/jobs/" + job_id, headers=foreign_headers).status_code == 404
    assert client.get(audio_path, headers=foreign_headers).status_code == 404
    assert client.put(root, headers=headers, json=dict(body, status="inactive")).status_code == 409
    assert (
        client.post("/api/v1/jobs/" + job_id + "/cancel", headers=headers).json()["data"]["processing_state"]
        == "cancelled"
    )
    assert client.post("/api/v1/jobs/" + job_id + "/cancel", headers=headers).status_code == 200
    assert (
        client.post(root + "/jobs", headers=job_headers, json=job_body).json()["data"]["processing_state"]
        == "cancelled"
    )
    remaining = [
        responses[0].json()["data"]["unique_id"],
        converted.json()["data"]["unique_id"],
        transcribed.json()["data"]["unique_id"],
    ]
    for identifier in remaining:
        assert client.post("/api/v1/jobs/" + identifier + "/cancel", headers=headers).status_code == 200
    assert client.put(root, headers=headers, json=dict(body, status="inactive")).status_code == 200
    assert (
        client.post(root + "/jobs", headers=dict(headers, **{"Idempotency-Key": "inactive"}), json=job_body).status_code
        == 404
    )
    assert client.get(audio_path, headers=headers).status_code == 404
    assert client.put(root, headers=headers, json=body).status_code == 200
    with container.engine.begin() as connection:
        connection.execute(text("UPDATE permissions SET status='inactive' WHERE entity IN ('dubbing_chat','dubbing')"))
    assert client.get(root, headers=headers).status_code == 403
    assert client.post(root + "/jobs", headers=job_headers, json=job_body).status_code == 403
    with container.engine.begin() as connection:
        connection.execute(text("UPDATE permissions SET status='active' WHERE entity IN ('dubbing_chat','dubbing')"))
