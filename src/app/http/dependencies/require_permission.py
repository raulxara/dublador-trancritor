from fastapi import Request

from app.exceptions.authentication_error import AuthenticationError
from app.exceptions.authorization_error import AuthorizationError
from app.services.actor.authorized_actor import AuthorizedActor


class RequirePermission:
    """Requires trusted actor populated by the protected router authentication dependency."""

    def __init__(self, permission: str) -> None:
        self.permission = permission

    def __call__(self, request: Request) -> AuthorizedActor:
        actor = getattr(request.state, "actor", None)
        if not isinstance(actor, AuthorizedActor):
            raise AuthenticationError()
        if self.permission not in actor.permissions:
            raise AuthorizationError()
        return actor
