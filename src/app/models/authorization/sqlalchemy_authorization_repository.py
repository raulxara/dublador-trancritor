from sqlalchemy import Engine, text

from app.services.actor.authorized_actor import AuthorizedActor


class SqlAlchemyAuthorizationRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def find_actor(self, token_hash: str) -> AuthorizedActor | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    text("""
                SELECT c._id, c.office_id, c.user_id, c.profile_id
                FROM user_customers c
                JOIN users u ON u._id=c.user_id AND u.office_id=c.office_id AND u.status='active'
                JOIN offices o ON o._id=c.office_id AND o.status='active'
                JOIN profiles p ON p._id=c.profile_id AND p.office_id=c.office_id AND p.status='active'
                WHERE c.token=:token AND c.status='active' AND c.token_expires_at > UTC_TIMESTAMP()
            """),
                    {"token": token_hash},
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            permissions = connection.execute(
                text("""
                SELECT DISTINCT p.entity, p.action
                FROM user_position up
                JOIN positions pos ON pos._id=up.position_id AND pos.office_id=up.office_id
                JOIN position_permission pp ON pp.position_id=pos._id AND pp.office_id=up.office_id
                JOIN permissions p ON p._id=pp.permission_id
                WHERE up.user_id=:user AND up.office_id=:office
                  AND up.status='active' AND pos.status='active' AND pp.status='active' AND p.status='active'
                  AND (p.office_id IS NULL OR p.office_id=:office)
            """),
                {"user": row["user_id"], "office": row["office_id"]},
            ).mappings()
            return AuthorizedActor(
                row["office_id"],
                row["user_id"],
                row["_id"],
                row["profile_id"],
                frozenset(f"{p['entity']}.{p['action']}" for p in permissions),
            )

    def replace_token(self, office_id, user_id, token_hash, expires_at):
        with self.engine.begin() as connection:
            result = connection.execute(
                text("""
                UPDATE user_customers c
                JOIN users u ON u._id=c.user_id AND u.office_id=c.office_id
                JOIN profiles p ON p._id=c.profile_id AND p.office_id=c.office_id
                SET c.token=:token, c.token_expires_at=:expires, c.updated_at=UTC_TIMESTAMP()
                WHERE c.office_id=:office AND c.user_id=:user
                  AND c.status='active' AND u.status='active' AND p.status='active'
            """),
                dict(office=office_id, user=user_id, token=token_hash, expires=expires_at),
            )
            return result.rowcount == 1

    def revoke_token(self, office_id, user_customer_id):
        with self.engine.begin() as connection:
            connection.execute(
                text("""UPDATE user_customers SET token=NULL, token_expires_at=NULL,
                updated_at=UTC_TIMESTAMP() WHERE office_id=:office AND _id=:customer"""),
                dict(office=office_id, customer=user_customer_id),
            )
