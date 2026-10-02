from uuid import UUID
from fastapi import Depends, Header, Request, Response
from app.http.dependencies.require_permission import RequirePermission
from app.http.requests.provision_company_request import ProvisionCompanyRequest
from app.services.actor.authorized_actor import AuthorizedActor
from app.exceptions.invalid_input_error import InvalidInputError
from app.use_cases.provision_company.dtos.provision_company_dto_in import ProvisionCompanyDtoIn
from app.use_cases.register_user.dtos.register_user_dto_in import RegisterUserDtoIn


def provision_company(body: ProvisionCompanyRequest,request: Request,response: Response,idempotency_key: str=Header(),actor: AuthorizedActor=Depends(RequirePermission("office.provision"))):
    try:
        key = str(UUID(idempotency_key))
    except ValueError:
        raise InvalidInputError() from None
    registration = RegisterUserDtoIn(actor,body.firstName,body.lastName,body.email,body.phone,body.username,body.password.get_secret_value(),body.documentType,body.documentValue)
    result = request.app.state.container.provision_company.exec(ProvisionCompanyDtoIn(registration,key,body.office.name,body.office.slug))
    response.headers["Cache-Control"] = "no-store"
    return {"success":True,"data":{"officeId":result.office_id,"userId":result.user_id,"userCustomerId":result.user_customer_id,"token":result.token,"expiresAt":result.expires_at,"assignedPositionSlug":"customer"}}
