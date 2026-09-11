import io
import math
import struct
import wave
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError


def wav_content():
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"".join(struct.pack("<h", int(1000 * math.sin(i / 10))) for i in range(48000)))
    return stream.getvalue()


def check_voice_endpoints(client, headers, container, office_id):
    assert client.get("/api/v1/voices").status_code == 401
    language = client.get("/api/v1/languages", headers=headers).json()["data"][0]["_id"]
    gender = client.get("/api/v1/gender", headers=headers).json()["data"][0]["_id"]
    payload = dict(name="Test voice", language_id=language, gender_id=gender, description="Example")
    assert (
        client.post("/api/v1/voices", headers=headers, json=dict(payload, office_id="foreign-office")).status_code
        == 422
    )
    assert client.post("/api/v1/voices", headers=headers, json=dict(payload, language_id="invalid")).status_code == 422
    response = client.post("/api/v1/voices", headers=headers, json=payload)
    assert response.status_code == 201, response.text
    voice = response.json()["data"]["unique_id"]
    root = "/api/v1/voices/" + voice
    assert client.get(root, headers=headers).json()["data"]["description"] == "Example"
    assert len(client.get("/api/v1/voices", headers=headers).json()["data"]) == 1
    assert client.get("/api/v1/voices?limit=101", headers=headers).status_code == 422
    with container.engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO voices (_id,office_id,name,language_id) VALUES "
                "('foreign-voice','foreign-office','Other',:language)"
            ),
            dict(language=language),
        )
    assert client.get("/api/v1/voices/foreign-voice", headers=headers).status_code == 404
    assert client.put("/api/v1/voices/foreign-voice", headers=headers, json=payload).status_code == 404
    assert len(client.get("/api/v1/voices", headers=headers).json()["data"]) == 1
    content = wav_content()
    upload_headers = dict(headers, **{"Content-Type": "audio/wav"})
    assert client.post(root + "/samples", headers=headers, content=content).status_code == 415
    assert client.post(root + "/samples", headers=upload_headers, content=b"invalid audio").status_code == 422
    assert (
        client.post("/api/v1/voices/foreign-voice/samples", headers=upload_headers, content=content).status_code == 404
    )
    response = client.post(root + "/samples", headers=upload_headers, content=content)
    assert response.status_code == 201, response.text
    sample = response.json()["data"]["unique_id"]
    assert response.json()["data"]["version"] == 1
    assert "storage_key" not in response.text
    assert client.get(root, headers=headers).json()["data"]["current_sample_id"] == sample
    audio_path = root + "/samples/" + sample + "/audio"
    assert client.get(audio_path, headers=headers).content == content
    assert client.get(audio_path, headers=headers).headers["cache-control"] == "no-store"
    assert client.get("/api/v1/voices/foreign-voice/samples/" + sample + "/audio", headers=headers).status_code == 404
    assert client.get(audio_path).status_code == 401
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(
            executor.map(lambda _: client.post(root + "/samples", headers=upload_headers, content=content), range(2))
        )
    assert all(r.status_code == 201 for r in responses), [r.text for r in responses]
    assert {r.json()["data"]["version"] for r in responses} == {2, 3}
    assert len(client.get(root + "/samples", headers=headers).json()["data"]) == 3
    other = client.post("/api/v1/voices", headers=headers, json=payload).json()["data"]["unique_id"]
    with pytest.raises(IntegrityError), container.engine.begin() as connection:
        connection.execute(
            text("UPDATE voices SET current_sample_id=:sample WHERE _id=:voice"), dict(sample=sample, voice=other)
        )
    assert client.put(root, headers=headers, json=dict(payload, status="inactive")).status_code == 200
    assert client.get(audio_path, headers=headers).status_code == 404
    assert client.post(root + "/samples", headers=upload_headers, content=content).status_code == 404
    assert client.put(root, headers=headers, json=dict(payload, status="active")).status_code == 200
    assert client.get(audio_path, headers=headers).status_code == 200
    with container.engine.begin() as connection:
        connection.execute(text("UPDATE permissions SET status='inactive' WHERE entity='voice'"))
    assert client.get(root, headers=headers).status_code == 403
    assert client.post(root + "/samples", headers=upload_headers, content=content).status_code == 403
    with container.engine.begin() as connection:
        connection.execute(text("UPDATE permissions SET status='active' WHERE entity='voice'"))
