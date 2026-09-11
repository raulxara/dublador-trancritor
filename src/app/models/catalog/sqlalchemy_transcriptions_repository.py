from dataclasses import asdict, fields

from sqlalchemy import text

from app.entities.transcription.transcription_entity import TranscriptionEntity


class SqlAlchemyTranscriptionsRepository:
    def __init__(self, connection):
        self.connection = connection

    @staticmethod
    def entity(row):
        return (
            TranscriptionEntity(
                **{f.name: row["_id" if f.name == "unique_id" else f.name] for f in fields(TranscriptionEntity)}
            )
            if row
            else None
        )

    def list(self, office_id, job_id, limit, offset):
        rows = self.connection.execute(
            text("""SELECT * FROM transcriptions WHERE office_id=:office AND job_id=:job
            AND status='active' ORDER BY version DESC LIMIT :limit OFFSET :offset"""),
            dict(office=office_id, job=job_id, limit=limit, offset=offset),
        ).mappings()
        return [self.entity(row) for row in rows]

    def latest(self, office_id, job_id):
        row = (
            self.connection.execute(
                text("""SELECT * FROM transcriptions WHERE office_id=:office AND job_id=:job
            ORDER BY version DESC LIMIT 1 FOR UPDATE"""),
                dict(office=office_id, job=job_id),
            )
            .mappings()
            .first()
        )
        return self.entity(row)

    def create(self, entity):
        self.connection.execute(
            text("""INSERT INTO transcriptions (_id,office_id,job_id,source_file_id,language_id,
            text,version,origin,edited_by_user_id,previous_transcription_id,status)
            VALUES (:unique_id,:office_id,:job_id,:source_file_id,:language_id,:text,:version,:origin,
            :edited_by_user_id,:previous_transcription_id,:status)"""),
            asdict(entity),
        )
