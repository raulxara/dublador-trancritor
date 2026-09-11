from decimal import Decimal

import pytest

from app.exceptions.invalid_input_error import InvalidInputError
from app.services.actor.authorized_actor import AuthorizedActor
from app.services.chat.hash_dubbing_request_service import HashDubbingRequestService
from app.use_cases.submit_dubbing_job.dtos.submit_dubbing_job_dto_in import SubmitDubbingJobDtoIn

ACTOR = AuthorizedActor("office", "user", "customer", None, frozenset({"dubbing.generate"}))


@pytest.mark.parametrize(
    "extra",
    [
        dict(idempotency_key=""),
        dict(idempotency_key="unicode-ç"),
        dict(speed=Decimal("NaN")),
        dict(pitch_semitones=Decimal("Infinity")),
        dict(speed=Decimal("0.501")),
        dict(input_text=""),
        dict(input_message_id="audio"),
        dict(operation="unsupported"),
        dict(preserve_timing=True),
    ],
)
def test_invalid_job_dto_is_rejected_without_http(extra):
    args = dict(actor=ACTOR, chat_id="chat", idempotency_key="key", operation="text_to_speech", input_text="Hello")
    args.update(extra)
    with pytest.raises(InvalidInputError):
        SubmitDubbingJobDtoIn(**args)


def test_idempotency_normalizes_decimal_and_text_without_actor_secrets():
    first = SubmitDubbingJobDtoIn(ACTOR, "chat", "key", "text_to_speech", input_text=" Hello ", speed=Decimal("1"))
    second = SubmitDubbingJobDtoIn(ACTOR, "chat", "key", "text_to_speech", input_text="Hello", speed=Decimal("1.00"))
    assert HashDubbingRequestService().exec(first) == HashDubbingRequestService().exec(second)
