from sqlalchemy import text

from app.entities.dubbing_job.job_output_entity import JobOutputEntity


class SqlAlchemyJobOutputsRepository:
    def __init__(self, connection):
        self.connection = connection

    def list(self, office_id, job_id):
        rows = self.connection.execute(
            text("""SELECT o._id AS unique_id,o.format,o.purpose,
            f.size_bytes,f.duration_ms,f.storage_key FROM dubbing_job_outputs o
            JOIN media_files f ON f._id=o.media_file_id AND f.office_id=o.office_id
            JOIN dubbing_jobs j ON j._id=o.job_id AND j.office_id=o.office_id
            WHERE o.office_id=:office AND o.job_id=:job AND o.status='active'
                AND f.status='active' AND f.storage_state='ready' AND j.status='active'
                AND j.processing_state='completed' ORDER BY o.id LIMIT 100 FOR SHARE"""),
            dict(office=office_id, job=job_id),
        ).mappings()
        return [JobOutputEntity(**row) for row in rows]
