from __future__ import annotations

import shutil

import pytest

from panelcast.config import Settings
from panelcast.paths import project_root


@pytest.fixture(scope="session")
def root():
    return project_root()


@pytest.fixture(scope="session")
def settings(root):
    return Settings(root=root)


@pytest.fixture(scope="session")
def spark():
    if shutil.which("java") is None:
        from panelcast.spark import _ensure_java

        _ensure_java()
    try:
        from panelcast.spark import get_spark

        session = get_spark(app_name="panelcast-tests", cores=2, driver_memory="2g")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Spark unavailable: {exc}")
    yield session
