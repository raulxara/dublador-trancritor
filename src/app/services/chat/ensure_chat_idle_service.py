from app.entities.dubbing_job.i_dubbing_job_repository import IJobRepository
from app.exceptions.conflict_error import ConflictError


class EnsureChatIdleService:
    def __init__(self, repository: IJobRepository):
        self.repository = repository

    def exec(self, office_id: str, chat_id: str) -> None:
        if self.repository.chat_has_work(office_id, chat_id):
            raise ConflictError()
