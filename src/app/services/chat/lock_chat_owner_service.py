from app.entities.dubbing_job.i_dubbing_job_repository import IJobRepository


class LockChatOwnerService:
    """Use one lock order for submissions and chat mutations before locking a chat."""

    def __init__(self, repository: IJobRepository):
        self.repository = repository

    def exec(self, office_id: str, owner_id: str) -> None:
        self.repository.lock_owner(office_id, owner_id)
