import json
from sqlalchemy import Engine, text
from app.entities.user.registered_user_entity import RegisteredUserEntity
from app.exceptions.authorization_error import AuthorizationError
from app.exceptions.conflict_error import ConflictError
from app.models.authorization.sqlalchemy_registration_repository import SqlAlchemyRegistrationRepository

class SqlAlchemyCompanyProvisioningRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def create(self, source: str, fingerprint: str, values: dict) -> RegisteredUserEntity:
        with self.engine.begin() as connection:
            if not connection.execute(text("SELECT _id FROM offices WHERE _id=:source AND status='active' FOR UPDATE"), {"source":source}).scalar_one_or_none():
                raise AuthorizationError()
            office = connection.execute(text("SELECT status,config FROM offices WHERE _id=:office"), values).mappings().first()
            if office:
                config = json.loads(office["config"] or "{}")
                if office["status"] != "active" or config.get("source") != source or config.get("fingerprint") != fingerprint:
                    raise ConflictError()
                row = connection.execute(text("SELECT * FROM user_customers WHERE office_id=:office AND _id=:customer AND status='active'"), {"office":values["office"],"customer":config["customer_id"]}).mappings().first()
                if not row:
                    raise ConflictError()
                connection.execute(text("UPDATE user_customers SET token=:token,token_expires_at=:expires WHERE _id=:customer"), values | {"customer":row["_id"]})
                return RegisteredUserEntity(values["office"],row["profile_id"],row["user_id"],row["_id"],row["user_level_id"],config["position_id"])
            connection.execute(text("INSERT INTO offices (_id,name,slug,language,currency) VALUES (:office,:office_name,:office_slug,'pt-BR','BRL')"), values)
            result = SqlAlchemyRegistrationRepository(self.engine).create_customer(values, connection)
            connection.execute(text("UPDATE offices SET config=:config WHERE _id=:office"), {"office":values["office"],"config":json.dumps({"source":source,"fingerprint":fingerprint,"customer_id":result.customer_id,"position_id":result.position_id})})
            return result
