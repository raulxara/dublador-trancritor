from uuid import uuid4

from sqlalchemy import Connection, text


class SqlAlchemyChatSourceRepository:
    def __init__(self, connection: Connection):
        self.connection = connection

    def sample(self, office_id, voice_id):
        row = (
            self.connection.execute(
                text("""SELECT s._id AS sample_id,v.language_id,f.storage_key
            FROM voices v JOIN voice_samples s ON s._id=v.current_sample_id AND s.voice_id=v._id
                AND s.office_id=v.office_id
            JOIN media_files f ON f._id=s.original_file_id AND f.office_id=s.office_id
            WHERE v.office_id=:office AND v._id=:voice AND v.status='active' AND s.status='active'
            AND s.validation_state='ready' AND f.status='active' AND f.storage_state='ready' FOR SHARE"""),
                dict(office=office_id, voice=voice_id),
            )
            .mappings()
            .first()
        )
        return dict(row) if row else None

    def language_active(self, unique_id):
        return (
            self.connection.execute(
                text("SELECT _id FROM languages WHERE _id=:id AND status='active' FOR SHARE"), dict(id=unique_id)
            ).first()
            is not None
        )

    def input_file(self, office_id, message_id):
        row = (
            self.connection.execute(
                text("""SELECT f._id AS file_id,f.storage_key FROM dubbing_message_files a
            JOIN media_files f ON f._id=a.media_file_id AND f.office_id=a.office_id
            WHERE a.office_id=:office AND a.message_id=:message AND a.purpose='source' AND a.status='active'
            AND f.status='active' AND f.storage_state='ready' FOR SHARE"""),
                dict(office=office_id, message=message_id),
            )
            .mappings()
            .first()
        )
        return dict(row) if row else None

    def create_input(self, office_id, user_id, message_id, metadata):
        values = dict(metadata, office=office_id, user=user_id, message=message_id, attachment=str(uuid4()))
        self.connection.execute(
            text("""INSERT INTO media_files (_id,office_id,created_by_user_id,original_name,
            storage_key,mime_type,size_bytes,duration_ms,sample_rate,channels,checksum,storage_state)
            VALUES (:file,:office,:user,'input.wav',:key,'audio/wav',:size,:duration,:rate,1,:checksum,'ready')"""),
            values,
        )
        self.connection.execute(
            text("""INSERT INTO dubbing_message_files (_id,office_id,message_id,media_file_id,purpose)
            VALUES (:attachment,:office,:message,:file,'source')"""),
            values,
        )
