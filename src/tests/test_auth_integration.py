"""Run only on an explicitly isolated, empty MySQL database."""

import hashlib
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.config.settings import Settings
from app.main import create_app
from app.providers.container import Container
from app.use_cases.bootstrap_access.dtos.bootstrap_access_dto_in import BootstrapAccessDtoIn


@pytest.mark.skipif(os.getenv("DUBBER_INTEGRATION") != "1", reason="requires isolated MySQL")
def test_mysql_authentication_lifecycle_and_scope():
    container = Container.build(Settings())
    dto = BootstrapAccessDtoIn(
        "Test office", "test-office", "Test", "User", "user@example.test", "test", "test-password-long"
    )
    try:
        result = container.bootstrap_access.exec(dto)
        with pytest.raises(ValueError):
            container.bootstrap_access.exec(dto)
        headers = {"Authorization": f"Bearer {result.token}"}
        with container.engine.connect() as connection:
            stored = connection.execute(text("SELECT token FROM user_customers")).scalar_one()
            assert stored == hashlib.sha256(result.token.encode()).hexdigest()
            assert stored != result.token
            foreign_keys = (
                connection.execute(
                    text(
                        (
                            "SELECT REFERENCED_COLUMN_NAME FROM information_schema.KEY_COLUMN_USAGE WHERE "
                            "TABLE_SCHEMA=DATABASE() AND REFERENCED_TABLE_NAME IS NOT NULL"
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert foreign_keys and "id" not in foreign_keys
        with TestClient(create_app(Settings())) as client:
            path = "/api/v1/auth/context"
            for auth in ["", "Basic value", "Bearer invalid", "Bearer " + ("a" * 64)]:
                assert client.get(path, headers={"Authorization": auth}).status_code == 401
            assert client.get(path, params={"token": result.token, "officeId": result.office_id}).status_code == 401
            response = client.get(path, headers=headers)
            assert response.status_code == 200
            assert response.json()["data"]["officeId"] == result.office_id
            assert result.token not in response.text
            assert response.headers["cache-control"] == "no-store"
            for table in ["offices", "users", "profiles", "user_customers"]:
                with container.engine.begin() as connection:
                    connection.execute(text(f"UPDATE {table} SET status='inactive'"))
                assert client.get(path, headers=headers).status_code == 401
                with container.engine.begin() as connection:
                    connection.execute(text(f"UPDATE {table} SET status='active'"))
            for table in ["user_position", "positions", "position_permission", "permissions"]:
                with container.engine.begin() as connection:
                    connection.execute(text(f"UPDATE {table} SET status='inactive'"))
                assert client.get(path, headers=headers).status_code == 403
                with container.engine.begin() as connection:
                    connection.execute(text(f"UPDATE {table} SET status='active'"))
            with container.engine.begin() as connection:
                connection.execute(text("UPDATE user_customers SET token_expires_at=UTC_TIMESTAMP()-INTERVAL 1 SECOND"))
            assert client.get(path, headers=headers).status_code == 401
            with container.engine.begin() as connection:
                connection.execute(text("UPDATE user_customers SET token_expires_at=UTC_TIMESTAMP()+INTERVAL 1 DAY"))
                connection.execute(
                    text(
                        (
                            "INSERT INTO offices (_id,name,slug,language,currency) VALUES "
                            "('foreign-office','Other','other','pt-BR','BRL')"
                        )
                    )
                )
                connection.execute(
                    text(
                        (
                            "INSERT INTO users (_id,office_id,user_type,username,password) VALUES "
                            "('foreign-user','foreign-office','office','other','unusable')"
                        )
                    )
                )
                connection.execute(
                    text(
                        (
                            "INSERT INTO profiles (_id,office_id,first_name,last_name,email) VALUES "
                            "('foreign-profile','foreign-office','Other','User','other@example.test')"
                        )
                    )
                )
                connection.execute(
                    text(
                        (
                            "INSERT INTO user_customers (_id,office_id,user_id,profile_id) VALUES "
                            "('foreign-customer','foreign-office','foreign-user','foreign-profile')"
                        )
                    )
                )
            assert client.post("/api/v1/users/foreign-user/access-token", headers=headers).status_code == 404
            with pytest.raises(IntegrityError), container.engine.begin() as connection:
                connection.execute(
                    text("UPDATE user_customers SET profile_id='foreign-profile' WHERE office_id=:office"),
                    {"office": result.office_id},
                )

            from tests.voice_scenarios import check_voice_endpoints

            check_voice_endpoints(client, headers, container, result.office_id)
            from tests.chat_scenarios import check_chat_endpoints

            check_chat_endpoints(client, headers, container, result.office_id)
            from tests.processing_scenarios import check_processing

            check_processing(client, headers, container, result.office_id)
            from tests.completion_scenarios import check_completion

            check_completion(client, headers, container, result.office_id)
            # A permission belonging to another office must never grant access even if linked.
            with container.engine.begin() as connection:
                connection.execute(
                    text("UPDATE permissions SET office_id='foreign-office' WHERE entity IN ('catalog','user')")
                )
            assert client.get(path, headers=headers).status_code == 403
            with container.engine.begin() as connection:
                connection.execute(
                    text("UPDATE permissions SET office_id=:office WHERE entity IN ('catalog','user')"),
                    {"office": result.office_id},
                )
            response = client.post(f"/api/v1/users/{result.user_id}/access-token", headers=headers)
            assert response.status_code == 201
            new_token = response.json()["data"]["token"]
            assert new_token != result.token
            assert client.get(path, headers=headers).status_code == 401
            new_headers = {"Authorization": f"Bearer {new_token}"}
            assert client.get(path, headers=new_headers).status_code == 200
            assert client.delete("/api/v1/auth/access-token", headers=new_headers).status_code == 204
            assert client.get(path, headers=new_headers).status_code == 401
    finally:
        container.close()
