from sqlalchemy import Engine, text

from app.entities.voice.voice_entity import VoiceEntity
from app.exceptions.invalid_input_error import InvalidInputError
from app.exceptions.resource_not_found_error import ResourceNotFoundError


class SqlAlchemyVoicesRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    @staticmethod
    def entity(row):
        return VoiceEntity(
            row["_id"],
            row["office_id"],
            row["name"],
            row["language_id"],
            row["gender_id"],
            row["current_sample_id"],
            row["status"],
            row["description"],
        )

    def find_by_unique_id(self, office_id, unique_id):
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    text("SELECT * FROM voices WHERE office_id=:office AND _id=:id"),
                    dict(office=office_id, id=unique_id),
                )
                .mappings()
                .first()
            )
            return self.entity(row) if row else None

    def list(self, office_id, limit, offset):
        with self.engine.connect() as connection:
            rows = connection.execute(
                text("""SELECT * FROM voices WHERE office_id=:office
                ORDER BY id DESC LIMIT :limit OFFSET :offset"""),
                dict(office=office_id, limit=limit, offset=offset),
            ).mappings()
            return [self.entity(row) for row in rows]

    def save(self, voice, create):
        with self.engine.begin() as connection:
            for table, identifier in [("languages", voice.language_id), ("gender", voice.gender_id)]:
                if (
                    identifier is not None
                    and not connection.execute(
                        text(f"SELECT _id FROM {table} WHERE _id=:id AND status='active' FOR SHARE"),
                        dict(id=identifier),
                    ).first()
                ):
                    raise InvalidInputError()
            values = dict(
                id=voice.unique_id,
                office=voice.office_id,
                name=voice.name,
                language=voice.language_id,
                gender=voice.gender_id,
                description=voice.description,
                status=voice.status,
            )
            if create:
                connection.execute(
                    text("""INSERT INTO voices (_id,office_id,name,language_id,gender_id,description)
                    VALUES (:id,:office,:name,:language,:gender,:description)"""),
                    values,
                )
            else:
                row = connection.execute(
                    text("SELECT _id FROM voices WHERE office_id=:office AND _id=:id FOR UPDATE"), values
                ).first()
                if row is None:
                    raise ResourceNotFoundError()
                connection.execute(
                    text("""UPDATE voices SET name=:name,language_id=:language,gender_id=:gender,
                    description=:description,status=:status,updated_at=UTC_TIMESTAMP()
                    WHERE office_id=:office AND _id=:id"""),
                    values,
                )
