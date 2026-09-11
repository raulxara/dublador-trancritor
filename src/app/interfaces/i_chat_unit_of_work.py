from typing import Protocol

from app.entities.dubbing_chat.i_dubbing_chat_repository import IChatRepository
from app.entities.dubbing_job.i_dubbing_job_repository import IJobRepository
from app.entities.dubbing_message.i_dubbing_message_repository import IMessageRepository
from app.entities.project_audio_link.i_project_audio_links_repository import IProjectAudioLinksRepository
from app.entities.tag.i_tags_repository import ITagsRepository
from app.entities.transcription.i_transcriptions_repository import ITranscriptionsRepository
from app.interfaces.i_chat_source_repository import IChatSourceRepository
from app.interfaces.i_job_outputs_repository import IJobOutputsRepository
from app.interfaces.i_unit_of_work import IUnitOfWork


class IChatUnitOfWork(IUnitOfWork, Protocol):
    outputs: IJobOutputsRepository
    tags: ITagsRepository
    project_links: IProjectAudioLinksRepository
    transcriptions: ITranscriptionsRepository
    chats: IChatRepository
    messages: IMessageRepository
    jobs: IJobRepository
    sources: IChatSourceRepository
