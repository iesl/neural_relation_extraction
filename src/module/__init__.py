import os
import logging

__all__ = ["logger"]

logger = logging.getLogger("BioRE")
LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logger.setLevel(LOGLEVEL)

console_logging_stream = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_logging_stream.setFormatter(formatter)
logger.addHandler(console_logging_stream)

