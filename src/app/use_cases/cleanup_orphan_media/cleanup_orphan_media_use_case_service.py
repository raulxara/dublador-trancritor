from app.interfaces.i_orphan_cleanup import IOrphanCleanup
from app.use_cases.cleanup_orphan_media.dtos.cleanup_orphan_media_dto_in import CleanupOrphanMediaDtoIn
from app.use_cases.cleanup_orphan_media.dtos.cleanup_orphan_media_dto_out import CleanupOrphanMediaDtoOut


class CleanupOrphanMediaUseCaseService:
    def __init__(self, service: IOrphanCleanup):
        self.service = service

    def exec(self, dto: CleanupOrphanMediaDtoIn) -> CleanupOrphanMediaDtoOut:
        return CleanupOrphanMediaDtoOut(self.service.exec(dto.minimum_age_seconds, dto.limit))
