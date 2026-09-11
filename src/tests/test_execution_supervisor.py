from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.exceptions.execution_timeout_error import ExecutionTimeoutError
from app.exceptions.lease_lost_error import LeaseLostError
from app.services.processing.subprocess_audio_execution import SubprocessAudioExecution


@pytest.mark.parametrize("cancel,timeout,error", [(True, 100, LeaseLostError), (False, 0, ExecutionTimeoutError)])
def test_supervisor_kills_process_group_on_cancel_or_deadline(tmp_path, cancel, timeout, error):
    settings = SimpleNamespace(model_root="/models", timeout_seconds=timeout, heartbeat_seconds=1)
    service = SubprocessAudioExecution(tmp_path, settings, stopping=lambda: cancel)
    process = SimpleNamespace(pid=9988, poll=lambda: None, wait=lambda: None)
    with patch("subprocess.Popen", return_value=process) as start, patch("os.killpg") as kill:
        with pytest.raises(error):
            service.supervise([], lambda: True)
        kill.assert_called_once()
        assert start.call_args.kwargs["start_new_session"] is True
        assert all("PASSWORD" not in key for key in start.call_args.kwargs["env"])
