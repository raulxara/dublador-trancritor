from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import text
from app.models.voice.sqlalchemy_default_voice_repository import SqlAlchemyDefaultVoiceRepository
from app.models.voice.sqlalchemy_voices_repository import SqlAlchemyVoicesRepository
from app.models.voice.sqlalchemy_sample_repository import SqlAlchemySampleRepository
from app.services.voice.default_voice_service import DefaultVoiceService
from app.services.voice.private_media_storage import PrivateMediaStorage
from tests.voice_scenarios import wav_content


def check_default_voice(container, office):
    with TemporaryDirectory() as root:
        source = Path(root) / "adam.wav"
        source.write_bytes(wav_content())
        service = DefaultVoiceService(SqlAlchemyDefaultVoiceRepository(container.engine), PrivateMediaStorage(root), source)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(service.ensure, [office] * 4))
        service.ensure("foreign-office")
        own = str(uuid5(NAMESPACE_URL, "siplug:default:adam:v1:" + office))
        other = str(uuid5(NAMESPACE_URL, "siplug:default:adam:v1:foreign-office"))
        voices = SqlAlchemyVoicesRepository(container.engine)
        assert voices.find_by_unique_id(office, own).current_sample_id
        assert voices.find_by_unique_id(office, other) is None
        samples = SqlAlchemySampleRepository(container.engine)
        assert len(samples.list(office, own, 10, 0)) == 1
        assert samples.list(office, other, 10, 0) == []
        with container.engine.connect() as connection:
            count = connection.execute(text("SELECT COUNT(*) FROM voices WHERE _id=:id"), dict(id=own)).scalar_one()
            assert count == 1
        # A later catalogue visit must not overwrite tenant customization.
        with container.engine.begin() as connection:
            connection.execute(text("UPDATE voices SET name='Custom Adam' WHERE _id=:id"), dict(id=own))
        service.ensure(office)
        assert voices.find_by_unique_id(office, own).name == "Custom Adam"

    from app.models.authorization.sqlalchemy_company_provisioning_repository import SqlAlchemyCompanyProvisioningRepository
    with container.engine.begin() as connection:
        position = connection.execute(text("SELECT _id FROM positions WHERE office_id=:office LIMIT 1"),dict(office=office)).scalar_one()
        SqlAlchemyCompanyProvisioningRepository.grant_service_permissions(connection, office, position)
        SqlAlchemyCompanyProvisioningRepository.grant_service_permissions(connection, office, position)
        names = connection.execute(text("SELECT slug FROM permissions WHERE office_id=:office AND _id IN (SELECT permission_id FROM position_permission WHERE position_id=:position)"),dict(office=office,position=position)).scalars().all()
        assert "catalog.read" in names and "dubbing.generate" in names
