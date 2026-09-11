import re
import shutil
import time
from pathlib import Path

from app.interfaces.i_media_inventory_repository import IMediaInventoryRepository
from app.services.media.media_publication_guard import MediaPublicationGuard


class OrphanCleanupService:
    """Delete only recognized old artifacts that have no database record."""

    def __init__(self, root, inventory: IMediaInventoryRepository):
        self.root = Path(root).resolve()
        self.inventory = inventory
        self.cursor = None

    def candidates(self):
        for directory in self.root.iterdir():
            if directory.is_symlink() or not directory.is_dir():
                continue
            if directory.name == ".attempts":
                for path in directory.iterdir():
                    if path.name.startswith("job-") and path.is_dir() and not path.is_symlink():
                        yield path, True
            elif re.fullmatch("[0-9a-f]{64}", directory.name):
                for path in directory.iterdir():
                    if (
                        not path.is_symlink()
                        and path.is_file()
                        and re.fullmatch(r"[0-9a-f-]{36}(?:\.[0-9a-f-]{36})?\.(?:wav|txt|tmp)", path.name)
                    ):
                        yield path, False

    def exec(self, minimum_age_seconds, limit):
        if minimum_age_seconds < 86400 or not 1 <= limit <= 1000:
            raise ValueError("Invalid cleanup policy")
        removed = 0
        with MediaPublicationGuard(self.root).hold(exclusive=True) as acquired:
            if not acquired:
                return 0
            if self.cursor is None:
                self.cursor = iter(self.candidates())
            for _ in range(limit):
                try:
                    path, directory = next(self.cursor)
                except (StopIteration, FileNotFoundError):
                    self.cursor = None
                    break
                if not path.exists() or path.is_symlink():
                    continue
                if time.time() - path.stat().st_mtime < minimum_age_seconds:
                    continue
                if self.inventory.referenced(path.relative_to(self.root).as_posix()):
                    continue
                if directory:
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed += 1
        return removed
