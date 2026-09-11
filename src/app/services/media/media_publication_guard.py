import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


class MediaPublicationGuard:
    """Shared publication lock; cleanup takes a nonblocking exclusive lock."""

    def __init__(self, root):
        self.root = Path(root)

    @contextmanager
    def hold(self, exclusive=False):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.root / ".publication.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        acquired = False
        try:
            try:
                fcntl.flock(descriptor, (fcntl.LOCK_EX | fcntl.LOCK_NB) if exclusive else fcntl.LOCK_SH)
                acquired = True
            except BlockingIOError:
                pass
            yield acquired
        finally:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
