from app.entities.dubbing_job.i_dubbing_job_repository import IJobRepository
from app.entities.dubbing_job.services.list_chat_jobs.dtos.list_chat_jobs_dto_in import ListChatJobsDtoIn
from app.entities.dubbing_job.services.list_chat_jobs.dtos.list_chat_jobs_dto_out import ListChatJobsDtoOut


class ListChatJobsService:
    def __init__(self, repository: IJobRepository):
        self.repository = repository

    def exec(self, dto: ListChatJobsDtoIn) -> ListChatJobsDtoOut:
        return ListChatJobsDtoOut(
            self.repository.list_chat(dto.office_id, dto.owner_id, dto.chat_id, dto.limit, dto.offset)
        )
