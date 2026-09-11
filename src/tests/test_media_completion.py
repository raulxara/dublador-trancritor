import os
import subprocess
import sys
import time
from uuid import uuid4

import pytest

from app.exceptions.invalid_input_error import InvalidInputError
from app.services.media.media_publication_guard import MediaPublicationGuard
from app.services.media.normalize_media_service import NormalizeMediaService
from app.services.media.orphan_cleanup_service import OrphanCleanupService
from tests.voice_scenarios import wav_content


@pytest.mark.parametrize(
    "kind,mime,args", [("mp3", "audio/mpeg", ["-c:a", "libmp3lame"]), ("mp4", "video/mp4", ["-c:a", "aac"])]
)
def test_real_mp3_mp4_normalization(tmp_path, kind, mime, args):
    source = tmp_path / "input.wav"
    source.write_bytes(wav_content())
    encoded = tmp_path / ("encoded." + kind)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source)]
    if kind == "mp4":
        command += ["-f", "lavfi", "-i", "color=c=black:s=32x32:d=3", "-shortest", "-c:v", "mpeg4"]
    subprocess.run(command + args + [str(encoded)], check=True)
    normalized = NormalizeMediaService().exec(encoded.read_bytes(), mime)
    assert normalized.startswith(b"RIFF")
    with pytest.raises(InvalidInputError):
        NormalizeMediaService().exec(b"invalid", mime)
    if mime == "video/mp4":
        video = tmp_path / "silent.mp4"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=32x32:d=3", "-an", str(video)],
            check=True,
        )
        with pytest.raises(InvalidInputError):
            NormalizeMediaService().exec(video.read_bytes(), mime)


def test_media_guard_and_orphan_reconciliation(tmp_path):
    root = tmp_path / "media"
    directory = root / ("a" * 64)
    directory.mkdir(parents=True)
    orphan = directory / (str(uuid4()) + ".wav")
    orphan.write_bytes(b"orphan")
    registered = directory / (str(uuid4()) + ".wav")
    registered.write_bytes(b"registered")
    young = directory / (str(uuid4()) + ".wav")
    young.write_bytes(b"new")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    symlink = directory / (str(uuid4()) + ".wav")
    symlink.symlink_to(outside)
    attempt = root / ".attempts" / "job-old"
    attempt.mkdir(parents=True)
    (attempt / "request.json").write_text("{}")
    for path in (orphan, registered, attempt):
        os.utime(path, (time.time() - 90000,) * 2)

    class Inventory:
        def referenced(self, key):
            return key == registered.relative_to(root).as_posix()

    service = OrphanCleanupService(root, Inventory())
    with MediaPublicationGuard(root).hold():
        assert service.exec(86400, 500) == 0 and orphan.exists()
    assert service.exec(86400, 500) == 2
    assert not orphan.exists() and not attempt.exists()
    assert registered.exists() and young.exists() and outside.exists() and symlink.is_symlink()
    orphan.write_bytes(b"orphan")
    os.utime(orphan, (time.time() - 90000,) * 2)

    class Unavailable:
        def referenced(self, key):
            raise RuntimeError("DB unavailable")

    with pytest.raises(RuntimeError):
        OrphanCleanupService(root, Unavailable()).exec(86400, 500)
    assert orphan.exists()


def test_publication_guard_is_shared_across_processes(tmp_path):
    code = (
        "from app.services.media.media_publication_guard import MediaPublicationGuard; import sys\n"
        "with MediaPublicationGuard(sys.argv[1]).hold(exclusive=True) as acquired: "
        "sys.exit(0 if acquired else 7)"
    )
    with MediaPublicationGuard(tmp_path).hold():
        assert subprocess.run([sys.executable, "-c", code, str(tmp_path)]).returncode == 7
    assert subprocess.run([sys.executable, "-c", code, str(tmp_path)]).returncode == 0
