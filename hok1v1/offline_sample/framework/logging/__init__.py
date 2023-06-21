from loguru import logger


INFO = 'INFO'
CRITICAL = 'CRITICAL'
ERROR = 'ERROR'
WARNING = 'WARNING'
INFO = 'INFO'
DEBUG = 'DEBUG'


def getLogger():
    return logger


def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)


def exception(msg, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)


def log(level, msg, *args, **kwargs):
    logger.log(level, msg, *args, **kwargs)
