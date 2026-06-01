"""Centralized logging configuration for the simulation.

Importing the engine must have **no** filesystem side effects, so the
warehouse log file is only created when an application explicitly calls
``configure_logging()`` at startup (typically from ``main.py``).
"""
import logging
from logging.handlers import RotatingFileHandler


def configure_logging(logfile='warehouse.log', level=logging.DEBUG,
                      console=True):
    """Configure application logging.

    Attaches a rotating file handler to the ``warehouse`` engine logger and,
    optionally, a console handler (INFO+) to the root logger so that
    user-facing status messages remain visible on stdout.

    Idempotent: repeated calls do not add duplicate handlers.

    Parameters
    ----------
    logfile : str
        Destination file for the engine's DEBUG-level event log.
    level : int
        Level for the engine file handler.
    console : bool
        When True, ensure an INFO-level console handler exists on the root
        logger for application status messages.
    """
    engine_log = logging.getLogger('warehouse')
    engine_log.setLevel(level)
    engine_log.propagate = False
    if not any(isinstance(h, RotatingFileHandler) for h in engine_log.handlers):
        handler = RotatingFileHandler(
            logfile, mode='a', maxBytes=5_000_000, backupCount=3,
            encoding='utf-8')
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s  %(levelname)-5s  %(message)s', datefmt='%H:%M:%S'))
        engine_log.addHandler(handler)

    if console:
        root = logging.getLogger()
        if root.level == logging.WARNING or root.level == 0:
            root.setLevel(logging.INFO)
        if not any(isinstance(h, logging.StreamHandler)
                   and not isinstance(h, RotatingFileHandler)
                   for h in root.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter('%(message)s'))
            root.addHandler(console_handler)

    return engine_log
