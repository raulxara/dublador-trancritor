from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.exceptions.authentication_error import AuthenticationError
from app.exceptions.authorization_error import AuthorizationError
from app.exceptions.conflict_error import ConflictError
from app.exceptions.invalid_input_error import InvalidInputError
from app.exceptions.resource_not_found_error import ResourceNotFoundError


def register_exception_handlers(application: FastAPI) -> None:
    def handler(status: int, code: str):
        async def handle(request: Request, exception: Exception) -> JSONResponse:
            return JSONResponse(
                status_code=status,
                content={"success": False, "code": code},
                headers={"WWW-Authenticate": "Bearer"} if status == 401 else None,
            )

        return handle

    for error, status, code in [
        (SQLAlchemyError, 503, "DUBBER_SERVICE_UNAVAILABLE"),
        (AuthenticationError, 401, "DUBBER_AUTHENTICATION_ERROR"),
        (AuthorizationError, 403, "DUBBER_AUTHORIZATION_ERROR"),
        (ResourceNotFoundError, 404, "DUBBER_RESOURCE_NOT_FOUND"),
        (ConflictError, 409, "DUBBER_CONFLICT"),
        (InvalidInputError, 422, "DUBBER_INVALID_INPUT"),
        (RequestValidationError, 422, "DUBBER_INVALID_INPUT"),
    ]:
        application.add_exception_handler(error, handler(status, code))
