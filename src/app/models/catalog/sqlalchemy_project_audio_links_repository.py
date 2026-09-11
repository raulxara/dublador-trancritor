from dataclasses import asdict

from sqlalchemy import text

from app.entities.project_audio_link.project_audio_link_entity import ProjectAudioLinkEntity


class SqlAlchemyProjectAudioLinksRepository:
    def __init__(self, connection):
        self.connection = connection

    @staticmethod
    def entity(row):
        return (
            ProjectAudioLinkEntity(
                row["_id"],
                row["office_id"],
                row["external_project_id"],
                row["media_file_id"],
                row["created_by_user_id"],
                row["status"],
            )
            if row
            else None
        )

    def owned_audio(self, office_id, owner_id, output_id):
        return self.connection.execute(
            text("""SELECT f._id FROM dubbing_job_outputs o
            JOIN media_files f ON f._id=o.media_file_id AND f.office_id=o.office_id
            JOIN dubbing_jobs j ON j._id=o.job_id AND j.office_id=o.office_id
            JOIN dubbing_messages m ON m._id=j.input_message_id AND m.office_id=j.office_id
            JOIN dubbing_chats c ON c._id=m.chat_id AND c.office_id=j.office_id
            WHERE o.office_id=:office AND o._id=:output AND o.status='active' AND o.purpose='audio'
            AND f.status='active' AND f.storage_state='ready' AND j.status='active' AND j.processing_state='completed'
            AND j.user_customer_id=:owner AND c.user_customer_id=:owner AND c.status='active' AND m.status='active'
            FOR SHARE"""),
            dict(office=office_id, owner=owner_id, output=output_id),
        ).scalar_one_or_none()

    def list(self, office_id, user_id, project_id, limit, offset):
        rows = self.connection.execute(
            text("""SELECT p.* FROM project_audio_links p
            JOIN media_files f ON f._id=p.media_file_id AND f.office_id=p.office_id
            WHERE p.office_id=:office AND p.created_by_user_id=:user AND p.external_project_id=:project
            AND p.status='active' AND f.status='active' AND f.storage_state='ready'
            ORDER BY p.id DESC LIMIT :limit OFFSET :offset"""),
            dict(office=office_id, user=user_id, project=project_id, limit=limit, offset=offset),
        ).mappings()
        return [self.entity(row) for row in rows]

    def find(self, office_id, project_id, media_id):
        row = (
            self.connection.execute(
                text("""SELECT * FROM project_audio_links WHERE office_id=:office
            AND external_project_id=:project AND media_file_id=:media FOR UPDATE"""),
                dict(office=office_id, project=project_id, media=media_id),
            )
            .mappings()
            .first()
        )
        return self.entity(row)

    def find_id(self, office_id, unique_id):
        row = (
            self.connection.execute(
                text("SELECT * FROM project_audio_links WHERE office_id=:office AND _id=:id FOR UPDATE"),
                dict(office=office_id, id=unique_id),
            )
            .mappings()
            .first()
        )
        return self.entity(row)

    def save(self, entity, create):
        query = (
            """INSERT INTO project_audio_links
            (_id,office_id,external_project_id,media_file_id,created_by_user_id,status)
            VALUES (:unique_id,:office_id,:external_project_id,:media_file_id,:created_by_user_id,:status)"""
            if create
            else """UPDATE project_audio_links SET status=:status,updated_at=UTC_TIMESTAMP()
            WHERE office_id=:office_id AND _id=:unique_id"""
        )
        self.connection.execute(text(query), asdict(entity))
