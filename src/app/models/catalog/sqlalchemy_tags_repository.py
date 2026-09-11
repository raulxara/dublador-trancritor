from dataclasses import asdict
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.entities.tag.tag_entity import TagEntity
from app.exceptions.conflict_error import ConflictError


class SqlAlchemyTagsRepository:
    def __init__(self, connection):
        self.connection = connection

    @staticmethod
    def entity(row):
        return TagEntity(row["_id"], row["office_id"], row["name"], row["slug"], row["status"]) if row else None

    def list(self, office_id, limit, offset):
        rows = self.connection.execute(
            text("SELECT * FROM tags WHERE office_id=:office ORDER BY id DESC LIMIT :limit OFFSET :offset"),
            dict(office=office_id, limit=limit, offset=offset),
        ).mappings()
        return [self.entity(row) for row in rows]

    def find(self, office_id, unique_id):
        row = (
            self.connection.execute(
                text("SELECT * FROM tags WHERE office_id=:office AND _id=:id FOR UPDATE"),
                dict(office=office_id, id=unique_id),
            )
            .mappings()
            .first()
        )
        return self.entity(row)

    def save(self, entity, create):
        query = (
            """INSERT INTO tags (_id,office_id,name,slug,status)
            VALUES (:unique_id,:office_id,:name,:slug,:status)"""
            if create
            else """UPDATE tags SET name=:name,slug=:slug,status=:status,updated_at=UTC_TIMESTAMP()
            WHERE office_id=:office_id AND _id=:unique_id"""
        )
        try:
            self.connection.execute(text(query), asdict(entity))
        except IntegrityError as error:
            if error.orig.args[0] == 1062:
                raise ConflictError() from None
            raise

    def voice_exists(self, office_id, voice_id):
        return (
            self.connection.execute(
                text("""SELECT _id FROM voices WHERE office_id=:office
            AND _id=:voice AND status='active' FOR UPDATE"""),
                dict(office=office_id, voice=voice_id),
            ).first()
            is not None
        )

    def list_voice(self, office_id, voice_id):
        rows = self.connection.execute(
            text("""SELECT t.* FROM tags t JOIN voice_tags vt
            ON vt.tag_id=t._id AND vt.office_id=t.office_id
            WHERE vt.office_id=:office AND vt.voice_id=:voice AND vt.status='active' AND t.status='active'
            ORDER BY t.slug"""),
            dict(office=office_id, voice=voice_id),
        ).mappings()
        return [self.entity(row) for row in rows]

    def replace_voice(self, office_id, voice_id, tag_ids):
        params = dict(office=office_id, voice=voice_id)
        self.connection.execute(
            text("""UPDATE voice_tags SET status='inactive',updated_at=UTC_TIMESTAMP()
            WHERE office_id=:office AND voice_id=:voice"""),
            params,
        )
        for tag in tag_ids:
            self.connection.execute(
                text("""INSERT INTO voice_tags (_id,office_id,voice_id,tag_id)
                VALUES (:id,:office,:voice,:tag) ON DUPLICATE KEY UPDATE status='active',updated_at=UTC_TIMESTAMP()"""),
                dict(params, id=str(uuid4()), tag=tag),
            )
