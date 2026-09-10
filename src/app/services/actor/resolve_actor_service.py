import hashlib
import re

from app.exceptions.authentication_error import AuthenticationError
from app.interfaces.i_authorization_repository import IAuthorizationRepository
from app.services.actor.authorized_actor import AuthorizedActor


class ResolveActorService:
    def __init__(self, repository: IAuthorizationRepository):
        self.repository = repository

    def exec(self, token: str) -> AuthorizedActor:
        if not re.fullmatch(r"[0-9a-f]{64}", token):
            raise AuthenticationError()
        actor = self.repository.find_actor(hashlib.sha256(token.encode()).hexdigest())
        if actor is None:
            raise AuthenticationError()
        return actor
