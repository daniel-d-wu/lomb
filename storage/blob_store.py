"""
Where the big files live: uploaded audio, the raw transcript JSON, Report 2
.xlsx exports. The database only stores a URI pointing here (audio_assets.
storage_uri, transcripts.raw_json_uri, report2_exports.storage_uri).

Same seam pattern as StorageRepository: lesson jobs are written against
BlobStore; LocalBlobStore (a folder) is for development and tests,
GCSBlobStore (gs://bucket/key, 2026-10-14) for production. No caller
changes between them.
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

    @abstractmethod
    def read_bytes(self, uri: str) -> bytes:
        """The whole object -- for small files like a Report 2 export."""


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

    def read_bytes(self, uri):
        return Path(uri).read_bytes()


class GCSBlobStore(BlobStore):
    """Google Cloud Storage; URIs are gs://bucket/key. Credentials come from
    the environment (the service account on Cloud Run) unless a client is
    passed in."""

    def __init__(self, bucket: str, *, client=None):
        if client is None:
            from google.cloud import storage
            client = storage.Client()
        self.bucket_name = bucket
        self._bucket = client.bucket(bucket)

    def _uri(self, key: str) -> str:
        return f"gs://{self.bucket_name}/{key}"

    def _blob(self, uri: str):
        prefix = f"gs://{self.bucket_name}/"
        if not uri.startswith(prefix):
            raise ValueError(f"not in bucket {self.bucket_name}: {uri!r}")
        return self._bucket.blob(uri[len(prefix):])

    def put_file(self, local_path, key):
        self._bucket.blob(key).upload_from_filename(str(local_path))
        return self._uri(key)

    def put_bytes(self, data, key):
        self._bucket.blob(key).upload_from_string(data)
        return self._uri(key)

    def fetch(self, uri, local_path):
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._blob(uri).download_to_filename(str(local_path))
        return local_path

    def exists(self, uri):
        return self._blob(uri).exists()

    def read_bytes(self, uri):
        return self._blob(uri).download_as_bytes()

    def browser_upload_url(self, key: str, *, content_type: str, size: int, origin: str | None) -> str:
        """A resumable-upload session the browser PUTs the file to directly
        (no size limit from the web API in between). The URL itself is the
        permission: it accepts this one object, and expires after a week if
        unused. `origin` lets that page's JavaScript make the cross-site PUT."""
        return self._bucket.blob(key).create_resumable_upload_session(content_type=content_type, size=size,
                                                                     origin=origin)
