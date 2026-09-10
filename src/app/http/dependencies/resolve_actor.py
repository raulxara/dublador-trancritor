from fastapi import Request

from app.exceptions.authentication_error import AuthenticationError
from app.services.actor.authorized_actor import AuthorizedActor


def resolve_actor(request: Request) -> AuthorizedActor:
    headers = request.headers.getlist("authorization")
    if len(headers) != 1:
        raise AuthenticationError()
    parts = headers[0].split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthenticationError()
    actor = request.app.state.container.resolve_actor.exec(parts[1])
    request.state.actor = actor
    return actor
