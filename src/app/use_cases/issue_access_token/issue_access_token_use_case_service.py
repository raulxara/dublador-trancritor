from app.exceptions.authorization_error import AuthorizationError
from app.services.actor.access_token_service import AccessTokenService
from app.use_cases.issue_access_token.dtos.issue_access_token_dto_in import IssueAccessTokenDtoIn
from app.use_cases.issue_access_token.dtos.issue_access_token_dto_out import IssueAccessTokenDtoOut


class IssueAccessTokenUseCaseService:
    def __init__(self, service: AccessTokenService):
        self.service = service

    def exec(self, dto: IssueAccessTokenDtoIn) -> IssueAccessTokenDtoOut:
        if "user.update" not in dto.actor.permissions:
            raise AuthorizationError()
        token, expires = self.service.issue(dto.actor.office_id, dto.user_id)
        return IssueAccessTokenDtoOut(dto.user_id, token, expires)
