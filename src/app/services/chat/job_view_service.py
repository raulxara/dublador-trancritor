from app.entities.dubbing_job.dubbing_job_entity import DubbingJobEntity


class JobViewService:
    @staticmethod
    def exec(job: DubbingJobEntity) -> dict:
        return dict(
            unique_id=job.unique_id,
            input_message_id=job.input_message_id,
            operation=job.operation,
            processing_state=job.processing_state,
            voice_sample_id=job.voice_sample_id,
            target_language_id=job.target_language_id,
            speed=str(job.speed),
            pitch_semitones=str(job.pitch_semitones),
            preserve_timing=bool(job.preserve_timing),
            status=job.status,
            output_message_id=job.output_message_id,
            error_code=job.error_code,
            attempts=job.attempts,
        )
