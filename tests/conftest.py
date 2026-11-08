import json

import pytest

from src import config


@pytest.fixture(scope="session")
def results():
    """Resultados del flujo completo. Si aún no existen, se generan una vez."""
    path = config.REPORTS / "metrics.json"
    if not path.exists():
        from src.run_all import run
        run(make_figures=False)
    return json.loads(path.read_text())
