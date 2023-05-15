import time


def get_cur_time_str():
    """Get the current timestamp string like '20210618123423' which contains date and time information.

    Returns:
        str: Current timestamp string.
    """
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
