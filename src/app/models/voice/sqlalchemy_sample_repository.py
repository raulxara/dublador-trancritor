from sqlalchemy import Engine, text

from app.exceptions.resource_not_found_error import ResourceNotFoundError


class SqlAlchemySampleRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def register(self, office_id, voice_id, user_id, metadata):
        values = dict(metadata, office=office_id, voice=voice_id, user=user_id)
        with self.engine.begin() as connection:
            if not connection.execute(
                text("""SELECT _id FROM voices
                WHERE office_id=:office AND _id=:voice AND status='active' FOR UPDATE"""),
                values,
            ).first():
                raise ResourceNotFoundError()
            version = connection.execute(
                text("""SELECT COALESCE(MAX(version),0)+1 FROM voice_samples
                WHERE office_id=:office AND voice_id=:voice"""),
                values,
            ).scalar_one()
            values["version"] = version
            connection.execute(
                text("""INSERT INTO media_files
                (_id,office_id,created_by_user_id,original_name,storage_key,mime_type,size_bytes,
                 duration_ms,sample_rate,channels,checksum,storage_state)
                VALUES (:file,:office,:user,'sample.wav',:key,'audio/wav',
                        :size,:duration,:rate,1,:checksum,'ready')"""),
                values,
            )
            connection.execute(
                text("""INSERT INTO voice_samples
                (_id,office_id,voice_id,original_file_id,validation_state,version)
                VALUES (:sample,:office,:voice,:file,'ready',:version)"""),
                values,
            )
            connection.execute(
                text("""UPDATE voices SET current_sample_id=:sample,updated_at=UTC_TIMESTAMP()
                WHERE office_id=:office AND _id=:voice"""),
                values,
            )
        return dict(
            unique_id=metadata["sample"], voice_id=voice_id, version=version, validation_state="ready", status="active"
        )

    def list(self, office_id, voice_id, limit, offset):
        with self.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    text("""SELECT _id AS unique_id,version,
                validation_state,status,created_at FROM voice_samples WHERE office_id=:office AND voice_id=:voice
                ORDER BY version DESC LIMIT :limit OFFSET :offset"""),
                    dict(office=office_id, voice=voice_id, limit=limit, offset=offset),
                ).mappings()
            ]

    def find_file(self, office_id, voice_id, sample_id):
        with self.engine.connect() as connection:
            return connection.execute(
                text("""SELECT f.storage_key FROM voice_samples s
                JOIN media_files f ON f._id=s.original_file_id AND f.office_id=s.office_id
                JOIN voices v ON v._id=s.voice_id AND v.office_id=s.office_id
                WHERE s.office_id=:office AND s.voice_id=:voice AND s._id=:sample
                AND s.status='active' AND s.validation_state='ready'
                AND f.status='active' AND f.storage_state='ready' AND v.status='active'"""),
                dict(office=office_id, voice=voice_id, sample=sample_id),
            ).scalar_one_or_none()
