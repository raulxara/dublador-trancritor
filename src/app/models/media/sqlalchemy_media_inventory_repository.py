from sqlalchemy import text


class SqlAlchemyMediaInventoryRepository:
    def __init__(self, engine):
        self.engine = engine

    def referenced(self, key):
        with self.engine.connect() as connection:
            return (
                connection.execute(
                    text("SELECT _id FROM media_files WHERE storage_key=:key LIMIT 1"), dict(key=key)
                ).first()
                is not None
            )
