from alembic import context

from app.config.settings import Settings
from app.providers.container import Container

container = Container.build(Settings())
try:
    with container.engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()
finally:
    container.close()
