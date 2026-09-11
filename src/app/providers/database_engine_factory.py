from sqlalchemy import URL, create_engine

from app.config.settings import Settings


class DatabaseEngineFactory:
    @staticmethod
    def build(settings: Settings):
        url = URL.create(
            "mysql+pymysql",
            username=settings.db_username,
            password=settings.db_password.get_secret_value(),
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_database,
        )
        engine = create_engine(
            url,
            pool_pre_ping=True,
            hide_parameters=True,
            pool_size=5,
            max_overflow=5,
            connect_args={"connect_timeout": 2, "read_timeout": 2, "write_timeout": 2},
        )
        return engine
