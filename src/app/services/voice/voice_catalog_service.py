from app.interfaces.i_voice_catalog_repository import IVoiceCatalogRepository


class VoiceCatalogService:
    def __init__(self, repository: IVoiceCatalogRepository):
        self.repository = repository

    def list(self, catalog: str) -> list[dict]:
        return self.repository.list(catalog)
