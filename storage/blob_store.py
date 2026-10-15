"""
Where the big files live: uploaded audio, the raw transcript JSON, Report 2
.xlsx exports. The database only stores a URI pointing here (audio_assets.
storage_uri, transcripts.raw_json_uri, report2_exports.storage_uri).

Same seam pattern as StorageRepository: lesson jobs are written against
BlobStore; LocalBlobStore (a folder) is for development and tests, and a
Google Cloud Storage implementation (gs://bucket/key) is added for
production without changing any caller. Added 2026-10-11.
"""

import shutil
from abc import ABC, abstractmethod
from pathlib import Path


class BlobStore(ABC):
    @abstractmethod
    def put_file(self, local_path: str | Path, key: str) -> str:
        """Copy a local file to `key`; return its URI."""

    @abstractmethod
    def put_bytes(self, data: bytes, key: str) -> str:
        """Store bytes at `key`; return its URI."""

    @abstractmethod
    def fetch(self, uri: str, local_path: str | Path) -> Path:
        """Download `uri` to `local_path` (for processing); return the path."""

    @abstractmethod
    def exists(self, uri: str) -> bool: ...


class LocalBlobStore(BlobStore):
    """A folder on disk; URIs are absolute file paths."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"key escapes the store root: {key!r}")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put_file(self, local_path, key):
        dest = self._path(key)
        if Path(local_path).resolve() != dest:
            shutil.copyfile(local_path, dest)
        return str(dest)

    def put_bytes(self, data, key):
        dest = self._path(key)
        dest.write_bytes(data)
        return str(dest)

    def fetch(self, uri, local_path):
        local_path = Path(local_path)
        if Path(uri).resolve() != local_path.resolve():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(uri, local_path)
        return local_path

    def exists(self, uri):
        p = Path(uri)
        return p.is_file() and p.stat().st_size > 0
