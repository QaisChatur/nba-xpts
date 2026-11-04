import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".pytest_cache" / "mpl"))

from pipeline.env import ensure_xgboost_importable  # noqa: E402

ensure_xgboost_importable()

from pipeline import config  # noqa: E402

TARGET_DATE = "2026-10-28"
BLUEPRINT_GAMES = ["0022600132", "0022600133", "0022600134", "0022600136"]


def _have(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


@pytest.fixture(scope="session")
def have_blueprint():
    if not config.BLUEPRINT_GAMES_DIR.exists():
        pytest.skip("Blueprint sample not present")


@pytest.fixture(scope="session")
def have_pages():
    paths = [config.GAME_PAGES_DIR / f"{g}.next_data.json.gz" for g in BLUEPRINT_GAMES]
    if not _have(*paths):
        pytest.skip("cached NBA.com game pages not present")


@pytest.fixture(scope="session")
def have_model_a():
    if not _have(config.MODEL_A_PATH, config.MODEL_A_ASSETS_DIR / "assets_meta.json"):
        pytest.skip("Model A artifact or inference assets not present")


@pytest.fixture(scope="session")
def oct28_context(have_blueprint, have_pages):
    from pipeline.legacy.nightly import ingest, match
    return match(ingest(TARGET_DATE, only_games=BLUEPRINT_GAMES))
