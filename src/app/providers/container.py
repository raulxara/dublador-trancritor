from dataclasses import dataclass

from sqlalchemy import URL, Engine, create_engine

from app.config.settings import Settings
from app.models.authorization.sqlalchemy_authorization_repository import SqlAlchemyAuthorizationRepository
from app.models.authorization.sqlalchemy_bootstrap_repository import SqlAlchemyBootstrapRepository
from app.services.actor.access_token_service import AccessTokenService
from app.services.actor.bootstrap_access_service import BootstrapAccessService
from app.services.actor.resolve_actor_service import ResolveActorService
from app.services.database.sqlalchemy_database_probe import SqlAlchemyDatabaseProbe
from app.use_cases.bootstrap_access.bootstrap_access_use_case_service import BootstrapAccessUseCaseService
from app.use_cases.get_health.get_health_use_case_service import GetHealthUseCaseService
from app.use_cases.issue_access_token.issue_access_token_use_case_service import IssueAccessTokenUseCaseService


@dataclass(slots=True)
class Container:
    engine: Engine
    get_health: GetHealthUseCaseService

    resolve_actor: ResolveActorService
    access_token: AccessTokenService
    issue_access_token: IssueAccessTokenUseCaseService

    bootstrap_access: BootstrapAccessUseCaseService

    @classmethod
    def build(cls, settings: Settings) -> "Container":
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
        repository = SqlAlchemyAuthorizationRepository(engine)
        access_token = AccessTokenService(repository)
        return cls(
            engine=engine,
            get_health=GetHealthUseCaseService(SqlAlchemyDatabaseProbe(engine)),
            resolve_actor=ResolveActorService(repository),
            access_token=access_token,
            issue_access_token=IssueAccessTokenUseCaseService(access_token),
            bootstrap_access=BootstrapAccessUseCaseService(
                BootstrapAccessService(SqlAlchemyBootstrapRepository(engine))
            ),
        )

    def close(self) -> None:
        self.engine.dispose()
