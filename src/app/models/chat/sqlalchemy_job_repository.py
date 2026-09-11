from dataclasses import asdict, fields

from sqlalchemy import Connection, text

from app.entities.dubbing_job.dubbing_job_entity import DubbingJobEntity


class SqlAlchemyJobRepository:
    def __init__(self, connection: Connection):
        self.connection = connection

    @staticmethod
    def entity(row):
        return DubbingJobEntity(
            **{f.name: row["_id" if f.name == "unique_id" else f.name] for f in fields(DubbingJobEntity)}
        )

    def lock_owner(self, office_id, owner_id):
        self.connection.execute(
            text("""SELECT _id FROM user_customers
            WHERE office_id=:office AND _id=:owner FOR UPDATE"""),
            dict(office=office_id, owner=owner_id),
        ).first()

    def by_key(self, office_id, owner_id, key):
        row = (
            self.connection.execute(
                text("""SELECT j.*,m.chat_id FROM dubbing_jobs j JOIN dubbing_messages m
                ON m._id=j.input_message_id AND m.office_id=j.office_id
            WHERE j.office_id=:office AND j.user_customer_id=:owner AND j.idempotency_key=:key FOR UPDATE"""),
                dict(office=office_id, owner=owner_id, key=key),
            )
            .mappings()
            .first()
        )
        return self.entity(row) if row else None

    def find(self, office_id, unique_id):
        row = (
            self.connection.execute(
                text(
                    """SELECT j.*,m.chat_id FROM dubbing_jobs j JOIN dubbing_messages m
                ON m._id=j.input_message_id AND m.office_id=j.office_id
                WHERE j.office_id=:office AND j._id=:id FOR UPDATE"""
                ),
                dict(office=office_id, id=unique_id),
            )
            .mappings()
            .first()
        )
        return self.entity(row) if row else None

    def create(self, entity):
        self.connection.execute(
            text("""INSERT INTO dubbing_jobs
            (_id,office_id,user_customer_id,input_message_id,operation,input_text,input_file_id,voice_sample_id,
             target_language_id,speed,pitch_semitones,preserve_timing,parameters,idempotency_key,request_hash)
            VALUES (:unique_id,:office_id,:user_customer_id,:input_message_id,:operation,:input_text,:input_file_id,
             :voice_sample_id,:target_language_id,:speed,:pitch_semitones,:preserve_timing,:parameters,:idempotency_key,:request_hash)"""),
            asdict(entity),
        )

    def cancel(self, office_id, unique_id):
        self.connection.execute(
            text("""UPDATE dubbing_jobs SET processing_state='cancelled',
            finished_at=UTC_TIMESTAMP(),worker_token=NULL,locked_until=NULL,
            updated_at=UTC_TIMESTAMP() WHERE office_id=:office AND _id=:id
            AND processing_state IN ('queued','processing')"""),
            dict(office=office_id, id=unique_id),
        )

    def chat_has_work(self, office_id, chat_id):
        return (
            self.connection.execute(
                text("""SELECT j._id FROM dubbing_jobs j
            JOIN dubbing_messages m
                ON m._id=j.input_message_id AND m.office_id=j.office_id
            WHERE j.office_id=:office AND m.chat_id=:chat AND j.status='active'
            AND j.processing_state IN ('queued','processing') LIMIT 1"""),
                dict(office=office_id, chat=chat_id),
            ).first()
            is not None
        )

    def list_chat(self, office_id, owner_id, chat_id, limit, offset):
        rows = self.connection.execute(
            text("""SELECT j.*,m.chat_id FROM dubbing_jobs j
            JOIN dubbing_messages m ON m._id=j.input_message_id AND m.office_id=j.office_id
            WHERE j.office_id=:office AND j.user_customer_id=:owner AND m.chat_id=:chat AND j.status='active'
            ORDER BY j.id DESC LIMIT :limit OFFSET :offset"""),
            dict(office=office_id, owner=owner_id, chat=chat_id, limit=limit, offset=offset),
        ).mappings()
        return [self.entity(row) for row in rows]
