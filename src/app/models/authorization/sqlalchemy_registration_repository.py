from contextlib import nullcontext
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from app.entities.user.registered_user_entity import RegisteredUserEntity
from app.exceptions.conflict_error import ConflictError
from app.exceptions.resource_not_found_error import ResourceNotFoundError


class SqlAlchemyRegistrationRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def create_customer(self, values: dict, transaction=None) -> RegisteredUserEntity:
        values = dict(values)
        try:
            with (nullcontext(transaction) if transaction is not None else self.engine.begin()) as connection:
                # Serialize registrations within this tenant, including creation of its default role.
                office = connection.execute(
                    text("SELECT _id FROM offices WHERE _id=:office AND status='active' FOR UPDATE"), values
                ).scalar_one_or_none()
                if office is None:
                    raise ResourceNotFoundError()
                duplicate = connection.execute(
                    text(
                        "SELECT EXISTS(SELECT 1 FROM users WHERE office_id=:office AND username=:username) "
                        "OR EXISTS(SELECT 1 FROM profiles WHERE office_id=:office AND email=:email)"
                    ),
                    values,
                ).scalar()
                if duplicate:
                    raise ConflictError()
                position = (
                    connection.execute(
                        text("SELECT _id,status FROM positions WHERE office_id=:office AND slug='customer' FOR UPDATE"),
                        values,
                    )
                    .mappings()
                    .first()
                )
                if position:
                    if position["status"] != "active":
                        raise ConflictError()
                    values["position"] = position["_id"]
                else:
                    connection.execute(
                        text(
                            "INSERT INTO positions (_id,office_id,name,slug) "
                            "VALUES (:position,:office,'Cliente','customer')"
                        ),
                        values,
                    )
                for statement in [
                    "INSERT INTO profiles "
                    "(_id,office_id,first_name,last_name,email,phone,document_type,document_value) "
                    "VALUES (:profile,:office,:first_name,:last_name,:email,:phone,:document_type,:document_value)",
                    "INSERT INTO users (_id,office_id,user_type,username,password) "
                    "VALUES (:user,:office,'customer',:username,:password)",
                    "INSERT INTO user_position (_id,office_id,user_id,position_id) "
                    "VALUES (:user_position,:office,:user,:position)",
                    "INSERT INTO user_customers "
                    "(_id,office_id,user_id,user_level_id,profile_id,token,token_expires_at) "
                    "VALUES (:customer,:office,:user,:user_position,:profile,:token,:expires)",
                ]:
                    connection.execute(text(statement), values)
        except IntegrityError:
            raise ConflictError() from None
        return RegisteredUserEntity(
            values["office"],
            values["profile"],
            values["user"],
            values["customer"],
            values["user_position"],
            values["position"],
        )
