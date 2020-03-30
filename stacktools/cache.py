import os
import json
import time
import logging
import hashlib
import pathlib

import numpy as np

from dbimage.io import read_dbim_from_fpath, write_array_to_fpath

CACHEPATH = pathlib.Path(os.path.expanduser("~/.cache/stacktools"))

logger = logging.getLogger("stacktools.cache")


def fn_caching_wrapper(func):

    def wrapped_func(*args, **kwargs):
        call_signature_b = bytearray(func.__name__, 'utf-8')
        for a in args:
            if isinstance(a, np.ndarray):
                call_signature_b += a.tobytes()
            else:
                call_signature_b += bytearray(a.__repr__(), 'utf-8')
                # raise Exception(f"Can't handle this type {type(a)}")

        for k, v in kwargs.items():
            call_signature_b += bytearray(f"{k}={v}", 'utf-8')

        fingerprint = hashlib.sha256(call_signature_b).hexdigest()

        cached_fpath = CACHEPATH / (fingerprint + ".dbim")
        info_fpath = CACHEPATH / (fingerprint + ".json")

        if os.path.isfile(cached_fpath):
            logger.debug(f"Found fingerprint in cache, loading {cached_fpath}")
            array = read_dbim_from_fpath(cached_fpath)
        else:
            logger.debug(f"No cache, running {func.__name__}")
            start_time = time.time()
            array = func(*args, **kwargs)
            end_time = time.time()
            CACHEPATH.mkdir(exist_ok=True, parents=True)
            write_array_to_fpath(cached_fpath, array)
            with open(info_fpath, 'w') as fh:
                json.dump(
                    {
                        "time": end_time-start_time,
                        "func_name": func.__name__
                    }, fh
                )

        return array

    return wrapped_func
