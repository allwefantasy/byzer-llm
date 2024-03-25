import logging
import sys

_FORMAT = "[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _reset_logger(log):
    for handler in log.handlers:
        handler.close()
        log.removeHandler(handler)
        del handler
    log.handlers.clear()
    log.propagate = False
    console_handle = logging.StreamHandler(sys.stdout)
    console_handle.setFormatter(
        logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
    )
    file_handle = logging.FileHandler("byzerllm.log", encoding="utf-8")
    file_handle.setFormatter(
        logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
    )
    log.addHandler(file_handle)
    log.addHandler(console_handle)


def _get_logger():
    log = logging.getLogger("byzerllm")
    _reset_logger(log)
    log.setLevel(logging.INFO)
    return log


# 日志句柄
logger = _get_logger()
