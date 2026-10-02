import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from app.interfaces.i_company_provisioning_repository import ICompanyProvisioningRepository
from app.use_cases.provision_company.dtos.provision_company_dto_in import ProvisionCompanyDtoIn
from app.use_cases.register_user.dtos.register_user_dto_out import RegisterUserDtoOut

class ProvisionCompanyService:
    def __init__(self, repository: ICompanyProvisioningRepository):
        self.repository = repository

    def exec(self, dto: ProvisionCompanyDtoIn) -> RegisterUserDtoOut:
        user = dto.registration
        token = secrets.token_hex(32)
        salt = secrets.token_hex(16)
        password = hashlib.scrypt(user.password.encode(),salt=salt.encode(),n=16384,r=8,p=1).hex()
        expires = datetime.now(timezone.utc)+timedelta(days=30)
        values = {key:str(uuid4()) for key in ["profile","user","customer","position","user_position"]}
        values.update(office=dto.key,office_name=dto.name,office_slug=dto.slug+"-"+dto.key[:8],first_name=user.first_name,last_name=user.last_name,email=user.email,phone=user.phone,username=user.username,document_type=user.document_type,document_value=user.document_value,password=f"scrypt$16384$8$1${salt}${password}",token=hashlib.sha256(token.encode()).hexdigest(),expires=expires.replace(tzinfo=None))
        fingerprint = hashlib.sha256(json.dumps([dto.name,dto.slug,user.email,user.username,user.first_name,user.last_name,user.phone,user.document_type,user.document_value]).encode()).hexdigest()
        result = self.repository.create(user.actor.office_id,fingerprint,values)
        return RegisterUserDtoOut(result.office_id,result.profile_id,result.user_id,result.customer_id,result.user_position_id,result.position_id,expires,token)
