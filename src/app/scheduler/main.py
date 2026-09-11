import signal
from threading import Event

from app.config.settings import Settings
from app.config.worker_settings import WorkerSettings
from app.providers.scheduler_container import SchedulerContainer
from app.use_cases.cleanup_orphan_media.dtos.cleanup_orphan_media_dto_in import CleanupOrphanMediaDtoIn
from app.use_cases.recover_expired_jobs.dtos.recover_expired_jobs_dto_in import RecoverExpiredJobsDtoIn
from app.workers.health import ProcessHealth


def main():
    stop = Event()
    health = ProcessHealth("scheduler")
    health.clear()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    policy = WorkerSettings()
    container = SchedulerContainer.build(Settings())
    try:
        while not stop.is_set():
            try:
                result = container.recover.exec(RecoverExpiredJobsDtoIn(policy.max_attempts))
                cleaned = container.cleanup.exec(CleanupOrphanMediaDtoIn())
                if cleaned.removed:
                    print("orphan_files_removed=" + str(cleaned.removed), flush=True)
                health.touch()
                if result.recovered:
                    print("recovered=" + str(result.recovered), flush=True)
            except Exception:
                health.clear()
                print("scheduler_cycle_failed", flush=True)
            stop.wait(10)
    finally:
        health.clear()
        container.engine.dispose()


if __name__ == "__main__":
    main()
