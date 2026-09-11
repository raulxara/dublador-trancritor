from app.entities.dubbing_job.services.claim_job.claim_job_service import ClaimJobService
from app.entities.dubbing_job.services.claim_job.dtos.claim_job_dto_in import ClaimJobDtoIn
from app.entities.dubbing_job.services.complete_execution.complete_execution_service import CompleteExecutionService
from app.entities.dubbing_job.services.complete_execution.dtos.complete_execution_dto_in import CompleteExecutionDtoIn
from app.entities.dubbing_job.services.fail_job.dtos.fail_job_dto_in import FailJobDtoIn
from app.entities.dubbing_job.services.fail_job.fail_job_service import FailJobService
from app.entities.dubbing_job.services.renew_job.dtos.renew_job_dto_in import RenewJobDtoIn
from app.entities.dubbing_job.services.renew_job.renew_job_service import RenewJobService
from app.entities.dubbing_job.services.resolve_execution.dtos.resolve_execution_dto_in import ResolveExecutionDtoIn
from app.entities.dubbing_job.services.resolve_execution.resolve_execution_service import ResolveExecutionService
from app.exceptions.execution_timeout_error import ExecutionTimeoutError
from app.exceptions.lease_lost_error import LeaseLostError
from app.interfaces.i_audio_execution import IAudioExecution
from app.use_cases.process_next_job.dtos.process_next_job_dto_in import ProcessNextJobDtoIn
from app.use_cases.process_next_job.dtos.process_next_job_dto_out import ProcessNextJobDtoOut


class ProcessNextJobUseCaseService:
    def __init__(
        self,
        claim: ClaimJobService,
        renew: RenewJobService,
        fail: FailJobService,
        resolve: ResolveExecutionService,
        complete: CompleteExecutionService,
        audio: IAudioExecution,
    ):
        self.claim, self.renew, self.fail = claim, renew, fail
        self.resolve, self.complete, self.audio = resolve, complete, audio

    def exec(self, dto: ProcessNextJobDtoIn) -> ProcessNextJobDtoOut:
        lease = self.claim.exec(ClaimJobDtoIn(dto.lease_seconds, dto.max_attempts)).data
        if lease is None:
            return ProcessNextJobDtoOut("idle")
        context = self.resolve.exec(ResolveExecutionDtoIn(lease)).data
        if context is None:
            self.fail.exec(FailJobDtoIn(lease, "INPUT_UNAVAILABLE"))
            return ProcessNextJobDtoOut("unavailable")

        def renew():
            return self.renew.exec(RenewJobDtoIn(lease, dto.lease_seconds)).data

        try:
            result = self.audio.run(context, renew)
        except LeaseLostError:
            return ProcessNextJobDtoOut("lease_lost")
        except ExecutionTimeoutError:
            self.fail.exec(FailJobDtoIn(lease, "EXECUTION_TIMEOUT"))
            return ProcessNextJobDtoOut("failed")
        except Exception:
            self.fail.exec(FailJobDtoIn(lease, "ENGINE_FAILED"))
            return ProcessNextJobDtoOut("failed")
        if not renew():
            return ProcessNextJobDtoOut("lease_lost")
        completed = self.complete.exec(CompleteExecutionDtoIn(lease, result)).data
        if not completed:
            self.fail.exec(FailJobDtoIn(lease, "INPUT_UNAVAILABLE"))
        return ProcessNextJobDtoOut("completed" if completed else "unavailable")
