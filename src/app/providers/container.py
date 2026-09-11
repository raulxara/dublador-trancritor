from dataclasses import dataclass

from sqlalchemy import Engine

from app.config.settings import Settings
from app.entities.voice.services.find_voice_by_unique_id.find_voice_by_unique_id_service import (
    FindVoiceByUniqueIdService,
)
from app.entities.voice.services.manage_voice.voice_service import VoiceService
from app.models.authorization.sqlalchemy_authorization_repository import SqlAlchemyAuthorizationRepository
from app.models.authorization.sqlalchemy_bootstrap_repository import SqlAlchemyBootstrapRepository
from app.models.chat.sqlalchemy_chat_unit_of_work import SqlAlchemyChatUnitOfWork
from app.models.voice.sqlalchemy_sample_repository import SqlAlchemySampleRepository
from app.models.voice.sqlalchemy_voice_catalog_repository import SqlAlchemyVoiceCatalogRepository
from app.models.voice.sqlalchemy_voices_repository import SqlAlchemyVoicesRepository
from app.providers.database_engine_factory import DatabaseEngineFactory
from app.services.actor.access_token_service import AccessTokenService
from app.services.actor.bootstrap_access_service import BootstrapAccessService
from app.services.actor.resolve_actor_service import ResolveActorService
from app.services.database.sqlalchemy_database_probe import SqlAlchemyDatabaseProbe
from app.services.voice.private_media_storage import PrivateMediaStorage
from app.services.voice.voice_catalog_service import VoiceCatalogService
from app.services.voice.voice_sample_service import VoiceSampleService
from app.use_cases.bootstrap_access.bootstrap_access_use_case_service import BootstrapAccessUseCaseService
from app.use_cases.cancel_dubbing_job.cancel_dubbing_job_use_case_service import CancelDubbingJobUseCaseService
from app.use_cases.create_dubbing_chat.create_dubbing_chat_use_case_service import CreateDubbingChatUseCaseService
from app.use_cases.create_project_audio_link.create_project_audio_link_use_case_service import (
    CreateProjectAudioLinkUseCaseService,
)
from app.use_cases.create_tag.create_tag_use_case_service import CreateTagUseCaseService
from app.use_cases.download_chat_audio.download_chat_audio_use_case_service import DownloadChatAudioUseCaseService
from app.use_cases.download_dubbing_output.download_dubbing_output_use_case_service import (
    DownloadDubbingOutputUseCaseService,
)
from app.use_cases.download_voice_sample.download_voice_sample_use_case_service import DownloadVoiceSampleUseCaseService
from app.use_cases.edit_transcription.edit_transcription_use_case_service import EditTranscriptionUseCaseService
from app.use_cases.get_dubbing_chat.get_dubbing_chat_use_case_service import GetDubbingChatUseCaseService
from app.use_cases.get_dubbing_job.get_dubbing_job_use_case_service import GetDubbingJobUseCaseService
from app.use_cases.get_health.get_health_use_case_service import GetHealthUseCaseService
from app.use_cases.get_voice.get_voice_use_case_service import GetVoiceUseCaseService
from app.use_cases.issue_access_token.issue_access_token_use_case_service import IssueAccessTokenUseCaseService
from app.use_cases.list_dubbing_chats.list_dubbing_chats_use_case_service import ListDubbingChatsUseCaseService
from app.use_cases.list_dubbing_jobs.list_dubbing_jobs_use_case_service import ListDubbingJobsUseCaseService
from app.use_cases.list_dubbing_messages.list_dubbing_messages_use_case_service import ListDubbingMessagesUseCaseService
from app.use_cases.list_dubbing_outputs.list_dubbing_outputs_use_case_service import ListDubbingOutputsUseCaseService
from app.use_cases.list_project_audio_links.list_project_audio_links_use_case_service import (
    ListProjectAudioLinksUseCaseService,
)
from app.use_cases.list_tags.list_tags_use_case_service import ListTagsUseCaseService
from app.use_cases.list_transcriptions.list_transcriptions_use_case_service import ListTranscriptionsUseCaseService
from app.use_cases.list_voice_catalog.list_voice_catalog_use_case_service import ListVoiceCatalogUseCaseService
from app.use_cases.list_voice_samples.list_voice_samples_use_case_service import ListVoiceSamplesUseCaseService
from app.use_cases.list_voice_tags.list_voice_tags_use_case_service import ListVoiceTagsUseCaseService
from app.use_cases.list_voices.list_voices_use_case_service import ListVoicesUseCaseService
from app.use_cases.manage_voice.manage_voice_use_case_service import ManageVoiceUseCaseService
from app.use_cases.register_voice_sample.register_voice_sample_use_case_service import RegisterVoiceSampleUseCaseService
from app.use_cases.set_voice_tags.set_voice_tags_use_case_service import SetVoiceTagsUseCaseService
from app.use_cases.submit_dubbing_job.submit_dubbing_job_use_case_service import SubmitDubbingJobUseCaseService
from app.use_cases.unlink_project_audio.unlink_project_audio_use_case_service import (
    UnlinkProjectAudioUseCaseService,
)
from app.use_cases.update_dubbing_chat.update_dubbing_chat_use_case_service import UpdateDubbingChatUseCaseService
from app.use_cases.update_tag.update_tag_use_case_service import UpdateTagUseCaseService
from app.use_cases.upload_chat_audio.upload_chat_audio_use_case_service import UploadChatAudioUseCaseService


@dataclass(slots=True)
class Container:
    list_dubbing_outputs: ListDubbingOutputsUseCaseService
    download_dubbing_output: DownloadDubbingOutputUseCaseService
    create_tag: CreateTagUseCaseService
    update_tag: UpdateTagUseCaseService
    list_tags: ListTagsUseCaseService
    list_voice_tags: ListVoiceTagsUseCaseService
    set_voice_tags: SetVoiceTagsUseCaseService
    create_project_audio_link: CreateProjectAudioLinkUseCaseService
    list_project_audio_links: ListProjectAudioLinksUseCaseService
    unlink_project_audio: UnlinkProjectAudioUseCaseService
    list_transcriptions: ListTranscriptionsUseCaseService
    edit_transcription: EditTranscriptionUseCaseService
    engine: Engine
    get_health: GetHealthUseCaseService

    resolve_actor: ResolveActorService
    access_token: AccessTokenService
    issue_access_token: IssueAccessTokenUseCaseService

    bootstrap_access: BootstrapAccessUseCaseService

    voice_service: VoiceService
    find_voice: FindVoiceByUniqueIdService
    voice_catalog: VoiceCatalogService
    manage_voice: ManageVoiceUseCaseService

    voice_samples: VoiceSampleService
    register_voice_sample: RegisterVoiceSampleUseCaseService

    list_voices: ListVoicesUseCaseService
    list_voice_samples: ListVoiceSamplesUseCaseService
    download_voice_sample: DownloadVoiceSampleUseCaseService
    list_voice_catalog: ListVoiceCatalogUseCaseService
    get_voice: GetVoiceUseCaseService

    create_dubbing_chat: CreateDubbingChatUseCaseService
    update_dubbing_chat: UpdateDubbingChatUseCaseService
    get_dubbing_chat: GetDubbingChatUseCaseService
    list_dubbing_chats: ListDubbingChatsUseCaseService
    list_dubbing_messages: ListDubbingMessagesUseCaseService
    submit_dubbing_job: SubmitDubbingJobUseCaseService
    get_dubbing_job: GetDubbingJobUseCaseService
    cancel_dubbing_job: CancelDubbingJobUseCaseService
    upload_chat_audio: UploadChatAudioUseCaseService
    download_chat_audio: DownloadChatAudioUseCaseService

    list_dubbing_jobs: ListDubbingJobsUseCaseService

    @classmethod
    def build(cls, settings: Settings) -> "Container":
        engine = DatabaseEngineFactory.build(settings)
        repository = SqlAlchemyAuthorizationRepository(engine)
        access_token = AccessTokenService(repository)
        voices = SqlAlchemyVoicesRepository(engine)
        voice_service = VoiceService(voices)
        samples = VoiceSampleService(
            voices, SqlAlchemySampleRepository(engine), PrivateMediaStorage(settings.media_root)
        )

        def chat_uow():
            return SqlAlchemyChatUnitOfWork(engine)

        chat_storage = PrivateMediaStorage(settings.media_root)
        return cls(
            edit_transcription=EditTranscriptionUseCaseService(chat_uow),
            list_transcriptions=ListTranscriptionsUseCaseService(chat_uow),
            unlink_project_audio=UnlinkProjectAudioUseCaseService(chat_uow),
            list_project_audio_links=ListProjectAudioLinksUseCaseService(chat_uow),
            create_project_audio_link=CreateProjectAudioLinkUseCaseService(chat_uow),
            set_voice_tags=SetVoiceTagsUseCaseService(chat_uow),
            list_voice_tags=ListVoiceTagsUseCaseService(chat_uow),
            list_tags=ListTagsUseCaseService(chat_uow),
            update_tag=UpdateTagUseCaseService(chat_uow),
            create_tag=CreateTagUseCaseService(chat_uow),
            download_dubbing_output=DownloadDubbingOutputUseCaseService(chat_uow, chat_storage),
            list_dubbing_outputs=ListDubbingOutputsUseCaseService(chat_uow, chat_storage),
            list_dubbing_jobs=ListDubbingJobsUseCaseService(chat_uow, chat_storage),
            create_dubbing_chat=CreateDubbingChatUseCaseService(chat_uow, chat_storage),
            update_dubbing_chat=UpdateDubbingChatUseCaseService(chat_uow, chat_storage),
            get_dubbing_chat=GetDubbingChatUseCaseService(chat_uow, chat_storage),
            list_dubbing_chats=ListDubbingChatsUseCaseService(chat_uow, chat_storage),
            list_dubbing_messages=ListDubbingMessagesUseCaseService(chat_uow, chat_storage),
            submit_dubbing_job=SubmitDubbingJobUseCaseService(chat_uow, chat_storage),
            get_dubbing_job=GetDubbingJobUseCaseService(chat_uow, chat_storage),
            cancel_dubbing_job=CancelDubbingJobUseCaseService(chat_uow, chat_storage),
            upload_chat_audio=UploadChatAudioUseCaseService(chat_uow, chat_storage),
            download_chat_audio=DownloadChatAudioUseCaseService(chat_uow, chat_storage),
            list_voices=ListVoicesUseCaseService(voice_service),
            list_voice_samples=ListVoiceSamplesUseCaseService(samples),
            download_voice_sample=DownloadVoiceSampleUseCaseService(samples),
            list_voice_catalog=ListVoiceCatalogUseCaseService(
                VoiceCatalogService(SqlAlchemyVoiceCatalogRepository(engine))
            ),
            get_voice=GetVoiceUseCaseService(FindVoiceByUniqueIdService(voices)),
            voice_samples=samples,
            register_voice_sample=RegisterVoiceSampleUseCaseService(samples),
            voice_service=voice_service,
            find_voice=FindVoiceByUniqueIdService(voices),
            voice_catalog=VoiceCatalogService(SqlAlchemyVoiceCatalogRepository(engine)),
            manage_voice=ManageVoiceUseCaseService(voice_service),
            engine=engine,
            get_health=GetHealthUseCaseService(SqlAlchemyDatabaseProbe(engine)),
            resolve_actor=ResolveActorService(repository),
            access_token=access_token,
            issue_access_token=IssueAccessTokenUseCaseService(access_token),
            bootstrap_access=BootstrapAccessUseCaseService(
                BootstrapAccessService(SqlAlchemyBootstrapRepository(engine))
            ),
        )

    def close(self) -> None:
        self.engine.dispose()
