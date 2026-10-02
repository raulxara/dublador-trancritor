import unittest
from dataclasses import replace
from app.services.actor.authorized_actor import AuthorizedActor
from app.services.actor.provision_company_service import ProvisionCompanyService
from app.use_cases.provision_company.provision_company_use_case_service import ProvisionCompanyUseCaseService
from app.use_cases.provision_company.dtos.provision_company_dto_in import ProvisionCompanyDtoIn
from app.use_cases.register_user.dtos.register_user_dto_in import RegisterUserDtoIn
from app.entities.user.registered_user_entity import RegisteredUserEntity
from app.exceptions.authorization_error import AuthorizationError
class Repository:
 def __init__(self): self.calls=[]
 def create(self,source,fingerprint,values):
  self.calls.append((source,fingerprint,values))
  return RegisteredUserEntity(values['office'],values['profile'],values['user'],values['customer'],values['user_position'],values['position'])
class ProvisionCompanyTest(unittest.TestCase):
 def setUp(self):
  self.repo=Repository();self.use_case=ProvisionCompanyUseCaseService(ProvisionCompanyService(self.repo),'platform')
  actor=AuthorizedActor('platform','admin','customer','profile',frozenset({'office.provision'}))
  self.dto=ProvisionCompanyDtoIn(RegisterUserDtoIn(actor,'Test','Customer','new@example.test','5511999999999','new@example.test','StrongPassword123!'),'00000000-0000-4000-8000-000000000001','New company','new-company')
 def test_dedicated_office_and_token_hash(self):
  import hashlib
  result=self.use_case.exec(self.dto);values=self.repo.calls[0][2]
  self.assertEqual(result.office_id,self.dto.key);self.assertNotEqual(result.office_id,'platform')
  self.assertEqual(values['token'],hashlib.sha256(result.token.encode()).hexdigest())
 def test_unapproved_source_is_rejected_before_write(self):
  dto=replace(self.dto,registration=replace(self.dto.registration,actor=replace(self.dto.registration.actor,office_id='foreign')))
  with self.assertRaises(AuthorizationError):self.use_case.exec(dto)
  self.assertEqual(self.repo.calls,[])
