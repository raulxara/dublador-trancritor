from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError


class SqlAlchemyDatabaseProbe:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def is_available(self) -> bool:
        try:
            with self.engine.connect() as connection:
                return connection.execute(text("SELECT 1")).scalar_one() == 1
        except SQLAlchemyError:
            # Never expose database URLs, passwords or driver exception messages over HTTP.
            return False
