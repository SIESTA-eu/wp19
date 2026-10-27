import logging
import os
from logging.handlers import TimedRotatingFileHandler

class _PrivateTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Opens every log file, including each one created at rotation, as
    owner-only (0600). Logs can hold request metadata and must not be
    readable by other users on the host."""

    def _open(self):
        stream = super()._open()
        os.chmod(self.baseFilename, 0o600)
        return stream


def setup_logger(name, log_dir="logs"):
    os.makedirs(log_dir, mode=0o700, exist_ok=True)
    # makedirs ignores mode for a directory that already exists.
    os.chmod(log_dir, 0o700)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_path = os.path.join(log_dir, f"{name}.log")

        handler = _PrivateTimedRotatingFileHandler(
            filename=log_path,
            when="midnight",       # Rota a medianoche
            interval=1,            # Every 1 day
            backupCount=7,         # Keep the last 7 days of logs
            encoding="utf-8",
            utc=False
        )
        # This suffix adds a date to rotated files: name.log.YYYY-MM-DD
        handler.suffix = "%Y-%m-%d"

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger

