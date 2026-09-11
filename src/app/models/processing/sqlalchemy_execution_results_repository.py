from uuid import uuid4

from sqlalchemy import Engine, text

from app.entities.dubbing_job.execution_input_entity import ExecutionInputEntity


class SqlAlchemyExecutionResultsRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    @staticmethod
    def context(connection, lease):
        row = (
            connection.execute(
                text("""SELECT j.*,m.chat_id,c.user_id,
            f.storage_key AS input_key,sf.storage_key AS sample_key,l.slug AS language
            FROM dubbing_jobs j
            JOIN user_customers c ON c._id=j.user_customer_id AND c.office_id=j.office_id AND c.status='active'
            JOIN users u ON u._id=c.user_id AND u.office_id=c.office_id AND u.status='active'
            JOIN profiles p ON p._id=c.profile_id AND p.office_id=c.office_id AND p.status='active'
            JOIN offices o ON o._id=j.office_id AND o.status='active'
            JOIN dubbing_messages m ON m._id=j.input_message_id AND m.office_id=j.office_id
                AND m.user_customer_id=j.user_customer_id AND m.status='active' AND m.role='user'
            JOIN dubbing_chats ch ON ch._id=m.chat_id AND ch.office_id=j.office_id
                AND ch.user_customer_id=j.user_customer_id AND ch.status='active'
            LEFT JOIN media_files f ON f._id=j.input_file_id AND f.office_id=j.office_id
                AND f.status='active' AND f.storage_state='ready'
            LEFT JOIN voice_samples s ON s._id=j.voice_sample_id AND s.office_id=j.office_id
                AND s.status='active' AND s.validation_state='ready'
            LEFT JOIN voices v ON v._id=s.voice_id AND v.office_id=s.office_id AND v.status='active'
            LEFT JOIN media_files sf ON sf._id=s.original_file_id AND sf.office_id=s.office_id
                AND sf.status='active' AND sf.storage_state='ready' AND v._id IS NOT NULL
            LEFT JOIN languages l ON l._id=j.target_language_id AND l.status='active'
            WHERE j._id=:id AND j.office_id=:office AND j.user_customer_id=:owner
                AND j.worker_token=:token AND j.processing_state='processing' AND j.status='active'
                AND j.locked_until>UTC_TIMESTAMP()
                AND (j.target_language_id IS NULL OR l._id IS NOT NULL)
                AND (j.operation='transcribe' OR sf._id IS NOT NULL)
                AND (j.operation='text_to_speech' OR (f._id IS NOT NULL AND EXISTS (
                    SELECT 1 FROM dubbing_message_files a WHERE a.office_id=j.office_id
                    AND a.message_id=m._id AND a.media_file_id=f._id AND a.purpose='source' AND a.status='active')))
            FOR SHARE"""),
                dict(id=lease.job_id, office=lease.office_id, owner=lease.owner_id, token=lease.token),
            )
            .mappings()
            .first()
        )
        return row

    @staticmethod
    def lock(connection, lease):
        connection.execute(
            text("SELECT _id FROM user_customers WHERE office_id=:office AND _id=:owner FOR UPDATE"),
            dict(office=lease.office_id, owner=lease.owner_id),
        ).first()
        return (
            connection.execute(
                text("""SELECT _id FROM dubbing_jobs WHERE _id=:id AND office_id=:office
            AND worker_token=:token AND processing_state='processing' AND status='active'
            AND locked_until>UTC_TIMESTAMP() FOR UPDATE"""),
                dict(id=lease.job_id, office=lease.office_id, token=lease.token),
            ).first()
            is not None
        )

    def resolve(self, lease):
        with self.engine.begin() as connection:
            if not self.lock(connection, lease):
                return None
            row = self.context(connection, lease)
            if row is None:
                return None
            return ExecutionInputEntity(
                row["office_id"],
                row["user_id"],
                row["chat_id"],
                row["input_message_id"],
                row["operation"],
                row["input_text"],
                row["input_key"],
                row["sample_key"],
                row["language"],
                float(row["speed"]),
                float(row["pitch_semitones"]),
                bool(row["preserve_timing"]),
            )

    def complete(self, lease, result):
        with self.engine.begin() as connection:
            if not self.lock(connection, lease):
                return False
            row = self.context(connection, lease)
            if row is None:
                return False
            message = str(uuid4())
            params = dict(
                office=lease.office_id,
                job=lease.job_id,
                message=message,
                user=row["user_id"],
                chat=row["chat_id"],
                input=row["input_message_id"],
                content=result.text,
            )
            connection.execute(
                text("""INSERT INTO dubbing_messages
                (_id,office_id,chat_id,role,message_type,content,reply_to_message_id)
                VALUES (:message,:office,:chat,'assistant','result',:content,:input)"""),
                params,
            )
            for file in result.files:
                values = dict(
                    params,
                    file=file.unique_id,
                    key=file.key,
                    size=file.size,
                    checksum=file.checksum,
                    duration=file.duration_ms,
                    rate=file.sample_rate,
                    channels=file.channels,
                    name="result." + file.format,
                    mime="audio/wav" if file.format == "wav" else "text/plain",
                    format=file.format,
                    purpose=file.purpose,
                    output=str(uuid4()),
                    attachment=str(uuid4()),
                )
                connection.execute(
                    text("""INSERT INTO media_files
                    (_id,office_id,created_by_user_id,original_name,storage_key,mime_type,size_bytes,
                    duration_ms,sample_rate,channels,checksum,storage_state)
                    VALUES (:file,:office,:user,:name,:key,:mime,:size,:duration,:rate,:channels,:checksum,'ready')"""),
                    values,
                )
                connection.execute(
                    text("""INSERT INTO dubbing_job_outputs
                    (_id,office_id,job_id,media_file_id,format,purpose)
                    VALUES (:output,:office,:job,:file,:format,:purpose)"""),
                    values,
                )
                connection.execute(
                    text("""INSERT INTO dubbing_message_files
                    (_id,office_id,message_id,media_file_id,purpose)
                    VALUES (:attachment,:office,:message,:file,'result')"""),
                    values,
                )
            if row["operation"] != "text_to_speech":
                transcription = str(uuid4())
                language = connection.execute(
                    text("""SELECT _id FROM languages WHERE status='active'
                    AND LOWER(REPLACE(slug,'_','-'))=:language ORDER BY id LIMIT 1"""),
                    dict(language=(result.language or "").lower()),
                ).scalar_one_or_none()
                connection.execute(
                    text("""INSERT INTO transcriptions
                    (_id,office_id,job_id,source_file_id,language_id,text,origin)
                    VALUES (:transcription,:office,:job,:source,:language,:content,'recognized')"""),
                    dict(params, transcription=transcription, source=row["input_file_id"], language=language),
                )
                for index, segment in enumerate(result.segments):
                    connection.execute(
                        text("""INSERT INTO dubbing_segments
                        (_id,office_id,job_id,transcription_id,`sequence`,source_start_ms,source_end_ms,
                        recognized_text,synthesis_text) VALUES (:id,:office,:job,:transcription,:sequence,
                        :start,:end,:recognized,:synthesis)"""),
                        dict(
                            params,
                            id=str(uuid4()),
                            transcription=transcription,
                            sequence=index,
                            start=segment.start_ms,
                            end=segment.end_ms,
                            recognized=segment.text,
                            synthesis=segment.text if row["operation"] == "speech_to_speech" else None,
                        ),
                    )
            published = connection.execute(
                text("""UPDATE dubbing_jobs SET processing_state='completed',
                output_message_id=:message,engine=:engine,model_version=:version,finished_at=UTC_TIMESTAMP(),
                worker_token=NULL,locked_until=NULL,error_code=NULL,error_message=NULL,updated_at=UTC_TIMESTAMP()
                WHERE _id=:job AND office_id=:office AND worker_token=:token AND locked_until>UTC_TIMESTAMP()"""),
                dict(params, token=lease.token, engine=result.engine, version=result.model_version),
            )
            if published.rowcount != 1:
                connection.rollback()
                return False
            return True
