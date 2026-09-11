import pytest

from app.exceptions.invalid_input_error import InvalidInputError
from app.services.voice.validate_wav_service import ValidateWavService
from tests.voice_scenarios import wav_content


@pytest.mark.parametrize("content", [b"", b"not audio", b"0" * (20 * 1024 * 1024 + 1)])
def test_reject_invalid_or_oversized_audio(content):
    with pytest.raises(InvalidInputError):
        ValidateWavService().exec(content)


def test_validate_real_frames_and_reject_truncation():
    content = wav_content()
    assert ValidateWavService().exec(content)["duration"] == 3000
    with pytest.raises(InvalidInputError):
        ValidateWavService().exec(content[:-10])
