from pydantic import BaseModel, ConfigDict, Field
from app.http.requests.register_user_request import RegisterUserRequest

class CompanyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1,max_length=250)
    slug: str = Field(min_length=1,max_length=100,pattern=r"^[a-z0-9-]+$")
    language: str = Field(pattern=r"^pt-BR$")
    currency: str = Field(pattern=r"^BRL$")

class ProvisionCompanyRequest(RegisterUserRequest):
    office: CompanyRequest
