import pytest

pytest.skip(
    "Neptune integration requires external service credentials; skipping in automated tests.",
    allow_module_level=True,
)
