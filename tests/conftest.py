"""Give tests their own database and vector store.

Without this, tests run against data/app.db and data/chroma -- they corrupt the
seeded catalog, they pollute each other, and sync_status() assertions become
dependent on whatever the last manual run left behind.

config is read lazily by db.conn() and vectors.collection(), so pointing those
paths at a tmpdir before the first connection is enough.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from app import config, db, vectors


@pytest.fixture(scope="session", autouse=True)
def isolated_stores():
    tmp = Path(tempfile.mkdtemp(prefix="reckon-test-"))
    config.DB_PATH = tmp / "test.db"
    config.CHROMA_PATH = tmp / "chroma"
    # embed_cache lives in the test DB too, so the first run of a test that
    # embeds does hit the API; repeat runs within a session are cached.
    db.init()
    yield tmp
    for c in (getattr(db._local, "conn", None),):  # noqa: SLF001
        if c:
            c.close()
    vectors._client = None      # noqa: SLF001
    vectors._collection = None  # noqa: SLF001
    shutil.rmtree(tmp, ignore_errors=True)
