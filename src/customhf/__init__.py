from logging import getLogger
from logging.config import fileConfig


_logger = getLogger("customhf")


def _try_setup_logging():
    try:
        fileConfig("logging.ini")
    except FileNotFoundError:
        pass  # if there is no config file, then just leave logging as is
