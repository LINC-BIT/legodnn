import csv
import os
import shutil
import json
import yaml
from typing import Any, List, Tuple, Union

from .log import logger
from .others import get_cur_time_str
from .file import ensure_dir


class CSVDataRecord:
    """Collect data into CSV file.
    Automatically backup existed file which has the same file name to avoid DATA LOST: 
    
    ```
    # data lost: all content in ./a-file-contains-important-data.csv will be 
    # flushed and unrecoverable if it's opened by 'w':
    with open('./a-file-contains-important-data.csv', 'w') as f:
        # do sth.
    ```
    
    Assuming a scene (actually it was my sad experience):
    - The code above is in the top of your experimental code,
    - And you've finished this experiment and collected the data into the CSV file.
    - After that, if you run this script file again accidentally, then all valuable data will be lost!
    
    :attr:`CSVDataRecord` makes this scene never happen again.
    """
    def __init__(self, file_path: str, header: List[str], backup=True):
        """Open the file and write CSV header into it.

        Args:
            file_path (str): Target CSV file path.
            header (List[str]): CSV header, like `['name', 'age', 'sex', ...]`.
            backup (bool, optional): If True, the existed file in :attr:`file_path` will be backup to `file_path + '.' + cur timestamp`. Defaults to True.
        """
        self.file_path = file_path
        self.header = header
        
        if backup and os.path.exists(file_path):
            backup_file_path = '{}.{}'.format(file_path, get_cur_time_str())
            shutil.copyfile(file_path, backup_file_path)
            logger.warn('csv file already exists! backup raw file to {}'.format(backup_file_path))

        ensure_dir(file_path)
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def write(self, data: Union[List[Any], Tuple[Any]]):
        """Write a row of data to file in :attr:`file_path`.

        Args:
            data (Union[List[Any], Tuple[Any]]):  A row of data, like `('ekko', 18, 'man')`.
        """
        assert len(data) == len(self.header)

        with open(self.file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)


def write_json(file_path: str, obj: Any, indent=2, backup=True):
    """Collect data into JSON file.
    Automatically backup existed file which has the same file name to avoid DATA LOST. (refers to :class:`CSVDataRecord`)

    Args:
        file_path (str): Target JSON file path.
        obj (Any): Collected data which can be serialized into JSON format.
        indent (int, optional): Keep indent to ensure readability. Defaults to 2.
        backup (bool, optional): If True, the existed file in :attr:`file_path` will be \
            backup to `file_path + '.' + cur timestamp`. Defaults to True.
    """
    if backup and os.path.exists(file_path):
        backup_file_path = '{}.{}'.format(file_path, get_cur_time_str())
        shutil.copyfile(file_path, backup_file_path)
        logger.warn('json file already exists! backup raw file to {}'.format(backup_file_path))
        
    with open(file_path, 'w') as f:
        obj_str = json.dumps(obj, indent=indent)
        f.write(obj_str)


def read_json(file_path: str):
    """Read JSON file.

    Args:
        file_path (str): Target JSON file path.

    Returns:
        Any: The object parsed from the target file.
    """
    with open(file_path, 'r') as f:
        return json.loads(f.read())


def read_yaml(file_path: str):
    """Read YAML file.

    Args:
        file_path (str): Target YAML file path.

    Returns:
        Any: The object parsed from the target file.
    """
    with open(file_path, 'r') as f:
        return yaml.load(f, yaml.Loader)
