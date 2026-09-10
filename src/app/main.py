from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config.settings import Settings
from app.http.exception_handlers import register_exception_handlers
from app.providers.container import Container
from app.routes.api import router


def create_app(settings: Settings | None = None) -> FastAPI:
    configuration = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.container = Container.build(configuration)
        try:
            yield
        finally:
            application.state.container.close()

    application = FastAPI(
        title=configuration.app_name,
        version="0.1.0",
        debug=False,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    register_exception_handlers(application)
    application.include_router(router, prefix="/api/v1")
    return application
