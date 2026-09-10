from app.interfaces.i_database_probe import IDatabaseProbe
from app.use_cases.get_health.dtos.get_health_dto_in import GetHealthDtoIn
from app.use_cases.get_health.dtos.get_health_dto_out import GetHealthDtoOut


class GetHealthUseCaseService:
    def __init__(self, database_probe: IDatabaseProbe) -> None:
        self.database_probe = database_probe

    def exec(self, dto_in: GetHealthDtoIn) -> GetHealthDtoOut:
        available = dto_in.check == "liveness" or self.database_probe.is_available()
        return GetHealthDtoOut(available=available, check=dto_in.check)
