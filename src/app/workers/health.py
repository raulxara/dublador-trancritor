import sys
import time
from pathlib import Path


class ProcessHealth:
    def __init__(self, name):
        self.path = Path("/tmp") / ("dubber-" + name + ".health")

    def touch(self):
        self.path.touch(mode=0o600)

    def clear(self):
        self.path.unlink(missing_ok=True)

    def healthy(self):
        return self.path.is_file() and time.time() - self.path.stat().st_mtime < 30


if __name__ == "__main__":
    sys.exit(0 if ProcessHealth(sys.argv[1]).healthy() else 1)
