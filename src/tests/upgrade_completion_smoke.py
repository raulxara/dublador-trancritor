"""Manual migration validation, exclusively on an empty disposable auth_test database."""

import os

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config.settings import Settings
from app.main import create_app
from app.providers.container import Container
from app.use_cases.bootstrap_access.dtos.bootstrap_access_dto_in import BootstrapAccessDtoIn


def main():
    settings = Settings()
    if os.getenv("DUBBER_INTEGRATION") != "1" or settings.db_database != "auth_test":
        raise SystemExit("Only disposable auth_test databases are allowed.")
    config = Config("alembic.ini")
    command.upgrade(config, "0004_processing_results")
    container = Container.build(settings)
    try:
        result = container.bootstrap_access.exec(
            BootstrapAccessDtoIn(
                "Upgrade", "upgrade", "Test", "User", "upgrade@example.test", "upgrade", "test-password-long"
            )
        )
        with container.engine.begin() as connection:
            connection.execute(
                text("""DELETE pp FROM position_permission pp JOIN permissions p ON p._id=pp.permission_id
                WHERE p.entity IN ('tag','project_audio','transcription')""")
            )
            connection.execute(text("DELETE FROM permissions WHERE entity IN ('tag','project_audio','transcription')"))
            token = connection.execute(text("SELECT token FROM user_customers")).scalar_one()
            connection.execute(
                text(
                    "INSERT INTO voices (_id,office_id,name,language_id) "
                    "SELECT 'upgrade-voice',:office,'Preserved',_id FROM languages LIMIT 1"
                ),
                dict(office=result.office_id),
            )
        for target in ["head", "0004_processing_results", "head"]:
            if target == "head":
                command.upgrade(config, target)
            else:
                command.downgrade(config, target)
            with container.engine.connect() as connection:
                assert connection.execute(text("SELECT token FROM user_customers")).scalar_one() == token
                assert (
                    connection.execute(text("SELECT name FROM voices WHERE _id='upgrade-voice'")).scalar_one()
                    == "Preserved"
                )
                assert connection.execute(text("SELECT COUNT(*) FROM users")).scalar_one() == 1
        with TestClient(create_app(settings)) as client:
            headers = {"Authorization": "Bearer " + result.token}
            assert client.get("/api/v1/tags", headers=headers).status_code == 200
            assert client.post("/api/v1/tags", headers=headers, json=dict(name="Test", slug="test")).status_code == 201
        with container.engine.connect() as connection:
            count = connection.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=DATABASE() AND "
                    "table_name!='alembic_version'"
                )
            ).scalar_one()
            assert count == 26, count
            assert (
                connection.execute(
                    text(
                        "SELECT COUNT(*) FROM information_schema.key_column_usage WHERE "
                        "table_schema=DATABASE() AND referenced_column_name='id'"
                    )
                ).scalar_one()
                == 0
            )
        print("Upgrade/downgrade/re-upgrade passed; credential and voice preserved; 26 domain tables.")
    finally:
        container.close()


if __name__ == "__main__":
    main()
