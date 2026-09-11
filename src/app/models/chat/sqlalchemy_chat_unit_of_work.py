from sqlalchemy import Engine

from app.models.catalog.sqlalchemy_project_audio_links_repository import SqlAlchemyProjectAudioLinksRepository
from app.models.catalog.sqlalchemy_tags_repository import SqlAlchemyTagsRepository
from app.models.catalog.sqlalchemy_transcriptions_repository import SqlAlchemyTranscriptionsRepository
from app.models.chat.sqlalchemy_chat_repository import SqlAlchemyChatRepository
from app.models.chat.sqlalchemy_chat_source_repository import SqlAlchemyChatSourceRepository
from app.models.chat.sqlalchemy_job_repository import SqlAlchemyJobRepository
from app.models.chat.sqlalchemy_message_repository import SqlAlchemyMessageRepository
from app.models.processing.sqlalchemy_job_outputs_repository import SqlAlchemyJobOutputsRepository


class SqlAlchemyChatUnitOfWork:
    def __init__(self, engine: Engine):
        self.engine = engine

    def __enter__(self):
        self.connection = self.engine.connect()
        self.transaction = self.connection.begin()
        self.outputs = SqlAlchemyJobOutputsRepository(self.connection)
        self.tags = SqlAlchemyTagsRepository(self.connection)
        self.project_links = SqlAlchemyProjectAudioLinksRepository(self.connection)
        self.transcriptions = SqlAlchemyTranscriptionsRepository(self.connection)
        self.chats = SqlAlchemyChatRepository(self.connection)
        self.messages = SqlAlchemyMessageRepository(self.connection)
        self.jobs = SqlAlchemyJobRepository(self.connection)
        self.sources = SqlAlchemyChatSourceRepository(self.connection)
        return self

    def commit(self):
        self.transaction.commit()

    def rollback(self):
        if self.transaction.is_active:
            self.transaction.rollback()

    def __exit__(self, exception_type, exception, traceback):
        try:
            self.rollback()
        finally:
            self.connection.close()
