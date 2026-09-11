from sqlalchemy import Engine, text


class SqlAlchemyVoiceCatalogRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def list(self, catalog):
        if catalog not in ("languages", "gender"):
            raise ValueError("Invalid catalog")
        with self.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    text(f"SELECT _id,name,slug FROM {catalog} WHERE status='active' ORDER BY name")
                ).mappings()
            ]
