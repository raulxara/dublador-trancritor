from fastapi import Request
from fastapi.responses import JSONResponse

from app.use_cases.get_health.dtos.get_health_dto_in import GetHealthDtoIn


def _response(request: Request, dto_in: GetHealthDtoIn) -> JSONResponse:
    result = request.app.state.container.get_health.exec(dto_in)
    return JSONResponse(
        status_code=200 if result.available else 503,
        content={
            "status": "ok" if result.available else "unavailable",
            "service": result.service,
            "check": result.check,
        },
    )


def get_health(request: Request) -> JSONResponse:
    return _response(request, GetHealthDtoIn(check="liveness"))


def get_readiness(request: Request) -> JSONResponse:
    return _response(request, GetHealthDtoIn(check="readiness"))
