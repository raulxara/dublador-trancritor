from uuid import uuid4

from sqlalchemy import Engine, text


class SqlAlchemyBootstrapRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def create_initial_access(self, values: dict) -> None:
        # MySQL named lock serializes concurrent CLI bootstraps before the first office exists.
        with self.engine.connect() as connection:
            acquired = connection.execute(text("SELECT GET_LOCK(CONCAT(DATABASE(), '-bootstrap'), 10)")).scalar()
            connection.commit()
            if acquired != 1:
                raise ValueError("Outro bootstrap está em execução.")
            try:
                with connection.begin():
                    if connection.execute(
                        text(
                            (
                                "SELECT EXISTS(SELECT 1 FROM offices) OR EXISTS(SELECT 1 FROM users) OR "
                                "EXISTS(SELECT 1 FROM user_customers)"
                            )
                        )
                    ).scalar():
                        raise ValueError("Bootstrap permitido somente em instalação vazia.")
                    for statement in [
                        (
                            "INSERT INTO offices (_id,name,slug,language,currency) VALUES "
                            "(:office,:office_name,:office_slug,'pt-BR','BRL')"
                        ),
                        (
                            "INSERT INTO profiles (_id,office_id,first_name,last_name,email) VALUES "
                            "(:profile,:office,:first_name,:last_name,:email)"
                        ),
                        (
                            "INSERT INTO users (_id,office_id,user_type,username,password) VALUES "
                            "(:user,:office,'office',:username,:password)"
                        ),
                        (
                            "INSERT INTO positions (_id,office_id,name,slug) VALUES "
                            "(:position,:office,'Administrador','admin')"
                        ),
                        (
                            "INSERT INTO user_position (_id,office_id,user_id,position_id) VALUES "
                            "(:user_position,:office,:user,:position)"
                        ),
                        (
                            "INSERT INTO user_customers "
                            "(_id,office_id,user_id,user_level_id,profile_id,token,token_expires_at) VALUES "
                            "(:customer,:office,:user,:user_position,:profile,:token,:expires)"
                        ),
                    ]:
                        connection.execute(text(statement), values)
                    for entity, action in [("catalog", "read"), ("user", "update")]:
                        permission = str(uuid4())
                        connection.execute(
                            text(
                                (
                                    "INSERT INTO permissions (_id,office_id,name,slug,entity,action) VALUES "
                                    "(:id,:office,:name,:name,:entity,:action)"
                                )
                            ),
                            dict(
                                id=permission,
                                office=values["office"],
                                name=f"{entity}.{action}",
                                entity=entity,
                                action=action,
                            ),
                        )
                        connection.execute(
                            text(
                                (
                                    "INSERT INTO position_permission (_id,office_id,position_id,permission_id) VALUES"
                                    " (:id,:office,:position,:permission)"
                                )
                            ),
                            dict(
                                id=str(uuid4()),
                                office=values["office"],
                                position=values["position"],
                                permission=permission,
                            ),
                        )
            finally:
                connection.execute(text("SELECT RELEASE_LOCK(CONCAT(DATABASE(), '-bootstrap'))"))
                connection.commit()
