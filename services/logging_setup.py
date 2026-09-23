"""Shared logging setup -- writes to both the console and data/studymate.log
so generation-pipeline diagnostics (e.g. how many flashcards survived each
filtering stage) are inspectable after a Streamlit run, not just scrolled
past in the terminal."""
import logging
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "studymate.log"
_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(LOG_PATH, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        _configured = True
    return logging.getLogger(name)
