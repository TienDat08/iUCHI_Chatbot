from functools import wraps
from loguru import logger
import sys
import time


logger.remove()
new_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
logger.add(sys.stderr, format=new_format)

def _patcher(record, prefix:str):
    record['name'] = prefix

def make_logger(prefix:str):
    r"""
    Create a custom logger for each service
    """
    return logger.patch(lambda record: _patcher(record, prefix))

def measure_time(logger):
    def main_decor(func):
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            _start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            logger.info("function: {}, processing time: {}".format(func.__name__,time.perf_counter() - _start_time))
            return result
        return wrapper

    return main_decor

