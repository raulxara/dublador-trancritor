from dataclasses import asdict

from sqlalchemy import Connection, text

from app.entities.dubbing_message.dubbing_message_entity import DubbingMessageEntity


class SqlAlchemyMessageRepository:
    def __init__(self, connection: Connection):
        self.connection = connection

    @staticmethod
    def entity(row):
        return DubbingMessageEntity(
            row["_id"],
            row["office_id"],
            row["chat_id"],
            row["user_customer_id"],
            row["role"],
            row["message_type"],
            row["content"],
            row["status"],
        )

    def create(self, entity):
        self.connection.execute(
            text("""INSERT INTO dubbing_messages
            (_id,office_id,chat_id,user_customer_id,role,message_type,content)
            VALUES (:unique_id,:office_id,:chat_id,:user_customer_id,:role,:message_type,:content)"""),
            asdict(entity),
        )
        self.connection.execute(
            text("""UPDATE dubbing_chats SET updated_at=UTC_TIMESTAMP()
            WHERE office_id=:office_id AND _id=:chat_id"""),
            asdict(entity),
        )

    def list(self, office_id, chat_id, limit, offset):
        rows = self.connection.execute(
            text("""SELECT * FROM dubbing_messages WHERE office_id=:office AND chat_id=:chat
            AND status='active' ORDER BY id ASC LIMIT :limit OFFSET :offset"""),
            dict(office=office_id, chat=chat_id, limit=limit, offset=offset),
        ).mappings()
        return [self.entity(row) for row in rows]

    def find(self, office_id, chat_id, unique_id):
        row = (
            self.connection.execute(
                text("""SELECT * FROM dubbing_messages
            WHERE office_id=:office AND chat_id=:chat AND _id=:id"""),
                dict(office=office_id, chat=chat_id, id=unique_id),
            )
            .mappings()
            .first()
        )
        return self.entity(row) if row else None
