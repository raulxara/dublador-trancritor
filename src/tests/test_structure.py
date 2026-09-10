import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.entities.voice.services.find_voice_by_unique_id.dtos.find_voice_by_unique_id_dto_in import (
    FindVoiceByUniqueIdDtoIn,
)
from app.entities.voice.services.find_voice_by_unique_id.find_voice_by_unique_id_service import (
    FindVoiceByUniqueIdService,
)
from app.entities.voice.voice_entity import VoiceEntity
from app.exceptions.resource_not_found_error import ResourceNotFoundError
from app.http.dependencies.require_permission import RequirePermission
from app.http.exception_handlers import register_exception_handlers
from app.main import create_app
from app.providers.container import Container
from app.services.actor.authorized_actor import AuthorizedActor
from app.use_cases.get_health.get_health_use_case_service import GetHealthUseCaseService


class FakeProbe:
    def __init__(self, available):
        self.available = available
        self.calls = 0

    def is_available(self):
        self.calls += 1
        return self.available


@pytest.mark.parametrize("database_available, expected", [(True, 200), (False, 503)])
def test_liveness_and_readiness_are_distinct(monkeypatch, database_available, expected):
    probe = FakeProbe(database_available)
    closed = []
    container = SimpleNamespace(get_health=GetHealthUseCaseService(probe), close=lambda: closed.append(True))
    monkeypatch.setattr(Container, "build", lambda settings: container)
    with TestClient(create_app(Settings(db_password="test-secret"))) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert probe.calls == 0
        response = client.get("/api/v1/ready")
        assert response.status_code == expected
        assert probe.calls == 1
        assert "test-secret" not in response.text
        assert client.get("/api/v1/voices").status_code == 404
        assert client.get("/docs").status_code == 404
    assert closed == [True]
    assert not any(name in sys.modules for name in ("torch", "tkinter", "TTS"))


@pytest.mark.parametrize("permissions, expected", [(None, 401), (frozenset(), 403), (frozenset({"voice.read"}), 200)])
def test_permission_guard_uses_only_server_actor(permissions, expected):
    application = FastAPI()
    register_exception_handlers(application)
    if permissions is not None:

        @application.middleware("http")
        async def server_actor(request, call_next):
            request.state.actor = AuthorizedActor("office", "user", "customer", None, permissions)
            return await call_next(request)

    @application.post("/guard", dependencies=[Depends(RequirePermission("voice.read"))])
    def endpoint():
        return {"success": True}

    with TestClient(application) as client:
        response = client.post(
            "/guard",
            json={"actor": {"permissions": ["voice.read"]}, "token": "secret"},
            headers={"Authorization": "Bearer secret"},
        )
        assert response.status_code == expected
        assert "secret" not in response.text
        if expected == 401:
            assert response.headers["www-authenticate"] == "Bearer"


class FakeVoiceRepository:
    def __init__(self, voice):
        self.voice = voice
        self.received = None

    def find_by_unique_id(self, office_id, unique_id):
        self.received = (office_id, unique_id)
        return self.voice


def test_entity_service_passes_scope_and_returns_dto():
    repo = FakeVoiceRepository(VoiceEntity("voice", "office", "Example", "language"))
    result = FindVoiceByUniqueIdService(repo).exec(FindVoiceByUniqueIdDtoIn("office", "voice"))
    assert repo.received == ("office", "voice")
    assert result.unique_id == "voice"
    assert not isinstance(result, VoiceEntity)


@pytest.mark.parametrize(
    "voice",
    [
        None,
        VoiceEntity("voice", "other-office", "Example", "language"),
        VoiceEntity("other-voice", "office", "Example", "language"),
    ],
)
def test_entity_service_rejects_foreign_or_incorrect_resource(voice):
    with pytest.raises(ResourceNotFoundError):
        FindVoiceByUniqueIdService(FakeVoiceRepository(voice)).exec(FindVoiceByUniqueIdDtoIn("office", "voice"))


def test_inner_layers_do_not_import_http_orm_or_engines():
    root = Path(__file__).resolve().parents[1] / "app"
    forbidden = {"fastapi", "sqlalchemy", "pydantic", "pydantic_settings", "torch", "TTS"}
    for folder in ("entities", "use_cases", "interfaces", "exceptions"):
        for file in (root / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(file.read_text())):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    assert module.split(".")[0] not in forbidden, (file, module)
                    assert not module.startswith(("app.http", "app.models", "app.providers")), (file, module)
