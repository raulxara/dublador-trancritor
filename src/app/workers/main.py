import signal
from threading import Event

from app.config.settings import Settings
from app.config.worker_settings import WorkerSettings
from app.providers.worker_container import WorkerContainer
from app.use_cases.process_next_job.dtos.process_next_job_dto_in import ProcessNextJobDtoIn
from app.workers.health import ProcessHealth


def main():
    stop = Event()
    health = ProcessHealth("worker")
    health.clear()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    policy = WorkerSettings()
    # Preflight does not report readiness before both cached models load successfully.
    container = WorkerContainer.build(Settings(), policy, stopping=stop.is_set)
    try:
        container.audio.preflight()
        container.audio.health_callback = health.touch
        while not stop.is_set():
            try:
                outcome = container.process.exec(ProcessNextJobDtoIn(policy.lease_seconds, policy.max_attempts))
                health.touch()
                if outcome.state != "idle":
                    print("job_state=" + outcome.state, flush=True)
            except Exception:
                health.clear()
                print("worker_cycle_failed", flush=True)
            stop.wait(policy.poll_seconds)
    finally:
        health.clear()
        container.engine.dispose()


if __name__ == "__main__":
    main()
