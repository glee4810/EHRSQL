import os
import time
import logging
from datetime import datetime, timedelta, timezone
timezone_adj = timezone(timedelta(hours=9)) # set timezone

time_format = '%Y-%m-%d %H:%M:%S'


def init_logger(output_path, args):
    # create output_dir
    os.makedirs(output_path, exist_ok=True)

    # file_name = args.mode+str(datetime.fromtimestamp(time.time(), timezone_adj).strftime('__%Y_%m_%d__%H_%M_%S'))+'.log'
    file_name = args.mode+'.log'
    file_path = os.path.join(output_path, file_name)

    if args.mode == 'train':
        log_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", time_format)
    else:
        log_formatter = logging.Formatter("%(message)s")


    # create file handler
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
    