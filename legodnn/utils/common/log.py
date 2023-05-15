import logging
import os
import sys

from .others import get_cur_time_str
from .file import ensure_dir


logger = logging.getLogger('zedl')
logger.setLevel(logging.DEBUG)
logger.propagate = False

formatter = logging.Formatter("%(asctime)s - %(filename)s[%(lineno)d] - %(levelname)s: %(message)s")
log_dir_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), './log')
if not os.path.exists(log_dir_path):
    os.mkdir(log_dir_path)

# file log
cur_time_str = get_cur_time_str()
log_file_path = os.path.join(log_dir_path, cur_time_str[0:8], cur_time_str[8:] + '.log')
ensure_dir(log_file_path)
file_handler = logging.FileHandler(log_file_path, mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# cmd log
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logging.getLogger('nni').setLevel(logging.ERROR)

# copy file content to log file
with open(os.path.abspath(sys.argv[0]), 'r') as f:
    content = f.read()
    logger.debug('entry file content: ---------------------------------')
    logger.debug('\n' + content)
    logger.debug('entry file content: ---------------------------------')
