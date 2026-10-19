from sqlalchemy import text


class SqlAlchemyDefaultVoiceRepository:
    def __init__(self, engine):
        self.engine = engine

    def exists(self, office_id, voice_id):
        with self.engine.connect() as connection:
            return connection.execute(text("SELECT _id FROM voices WHERE office_id=:office AND _id=:voice"),
                                      dict(office=office_id, voice=voice_id)).first() is not None

    def install(self, office_id, metadata):
        values = dict(metadata, office=office_id)
        with self.engine.begin() as connection:
            # Tenant row lock serializes concurrent first visits without global contention.
            if not connection.execute(text("SELECT _id FROM offices WHERE _id=:office FOR UPDATE"), values).first():
                raise ValueError("Unknown office")
            if connection.execute(text("SELECT _id FROM voices WHERE office_id=:office AND _id=:voice"), values).first():
                return
            values["language"] = connection.execute(text("SELECT _id FROM languages WHERE slug='pt-BR' AND status='active'")).scalar_one()
            connection.execute(text("""INSERT INTO voices (_id,office_id,name,language_id,description,config)
                VALUES (:voice,:office,'Adam',:language,'Voz padrão SiPlug','{"system_default":"adam-v1"}')"""), values)
            connection.execute(text("""INSERT INTO media_files
                (_id,office_id,original_name,storage_key,mime_type,size_bytes,duration_ms,sample_rate,channels,checksum,storage_state)
                VALUES (:file,:office,'adam.wav',:key,'audio/wav',:size,:duration,:rate,1,:checksum,'ready')"""), values)
            connection.execute(text("""INSERT INTO voice_samples
                (_id,office_id,voice_id,original_file_id,validation_state,version)
                VALUES (:sample,:office,:voice,:file,'ready',1)"""), values)
            connection.execute(text("UPDATE voices SET current_sample_id=:sample WHERE office_id=:office AND _id=:voice"), values)
