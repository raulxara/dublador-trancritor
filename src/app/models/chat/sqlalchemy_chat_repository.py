from dataclasses import asdict

from sqlalchemy import Connection, text

from app.entities.dubbing_chat.dubbing_chat_entity import DubbingChatEntity


class SqlAlchemyChatRepository:
    def __init__(self, connection: Connection):
        self.connection = connection

    @staticmethod
    def entity(row):
        return DubbingChatEntity(
            row["_id"], row["office_id"], row["user_customer_id"], row["title"], row["selected_voice_id"], row["status"]
        )

    def find(self, office_id, unique_id, lock=False):
        row = (
            self.connection.execute(
                text(
                    "SELECT * FROM dubbing_chats WHERE office_id=:office AND _id=:id" + (" FOR UPDATE" if lock else "")
                ),
                dict(office=office_id, id=unique_id),
            )
            .mappings()
            .first()
        )
        return self.entity(row) if row else None

    def list(self, office_id, owner_id, limit, offset):
        rows = self.connection.execute(
            text("""SELECT * FROM dubbing_chats
            WHERE office_id=:office AND user_customer_id=:owner ORDER BY id DESC LIMIT :limit OFFSET :offset"""),
            dict(office=office_id, owner=owner_id, limit=limit, offset=offset),
        ).mappings()
        return [self.entity(row) for row in rows]

    def save(self, entity, create):
        values = asdict(entity)
        if create:
            self.connection.execute(
                text("""INSERT INTO dubbing_chats (_id,office_id,user_customer_id,title,selected_voice_id)
                VALUES (:unique_id,:office_id,:user_customer_id,:title,:selected_voice_id)"""),
                values,
            )
        else:
            self.connection.execute(
                text("""UPDATE dubbing_chats SET title=:title,selected_voice_id=:selected_voice_id,
                status=:status,updated_at=UTC_TIMESTAMP() WHERE office_id=:office_id AND _id=:unique_id"""),
                values,
            )
