import asyncio
import functools
import logging
from typing import Tuple, Type


def retry(exceptions: Tuple[Type[Exception], ...], retry_count=3, retry_delay=1):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions:
                    logging.exception(f"Attempt #{attempt} out of {retry_count} has failed")
                    attempt += 1
                    if attempt == retry_count:
                        raise

                    await asyncio.sleep(retry_delay)
        return functools.update_wrapper(wrapper, func)
    return decorator

