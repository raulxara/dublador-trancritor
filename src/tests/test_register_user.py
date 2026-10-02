import hashlib
import unittest
from dataclasses import replace

from app.entities.user.registered_user_entity import RegisteredUserEntity
from app.exceptions.authorization_error import AuthorizationError
from app.services.actor.authorized_actor import AuthorizedActor
from app.services.actor.register_user_service import RegisterUserService
from app.use_cases.register_user.dtos.register_user_dto_in import RegisterUserDtoIn
from app.use_cases.register_user.register_user_use_case_service import RegisterUserUseCaseService


class Repository:
    def __init__(self):
        self.calls = []

    def create_customer(self, values):
        self.calls.append(values)
        return RegisteredUserEntity(values["office"], values["profile"], values["user"],
                                    values["customer"], values["user_position"], values["position"])


class RegisterUserTest(unittest.TestCase):
    def setUp(self):
        self.repo = Repository()
        self.use_case = RegisterUserUseCaseService(RegisterUserService(self.repo))
        self.dto = RegisterUserDtoIn(
            AuthorizedActor("office", "admin", "customer", "profile", frozenset({"user.register"})),
            "Test", "Customer", "customer@example.test", "5511999999999", "customer", "StrongPassword123!",
        )

    def test_no_permission_means_no_writes(self):
        actor = replace(self.dto.actor, permissions=frozenset())
        with self.assertRaises(AuthorizationError):
            self.use_case.exec(replace(self.dto, actor=actor))
        self.assertEqual(self.repo.calls, [])

    def test_foreign_office_means_no_writes(self):
        with self.assertRaises(AuthorizationError):
            self.use_case.exec(replace(self.dto, office_id="another-office"))
        self.assertEqual(self.repo.calls, [])

    def test_only_hashes_are_persisted_and_public_references_match(self):
        result = self.use_case.exec(self.dto)
        values = self.repo.calls[0]
        self.assertEqual(values["token"], hashlib.sha256(result.token.encode()).hexdigest())
        self.assertNotEqual(values["token"], result.token)
        _, n, r, parallel, salt, digest = values["password"].split("$")
        self.assertEqual(hashlib.scrypt(self.dto.password.encode(), salt=salt.encode(), n=int(n),
                                       r=int(r), p=int(parallel)).hex(), digest)
        self.assertEqual(result.office_id, self.dto.actor.office_id)
        self.assertEqual(result.user_id, values["user"])
        self.assertEqual(result.user_customer_id, values["customer"])
        self.assertNotIn(self.dto.password, repr(self.dto))
        self.assertNotIn(result.token, repr(result))

    def test_each_registration_uses_new_salt_and_token(self):
        first = self.use_case.exec(self.dto)
        second = self.use_case.exec(self.dto)
        self.assertNotEqual(first.token, second.token)
        self.assertNotEqual(self.repo.calls[0]["password"], self.repo.calls[1]["password"])
