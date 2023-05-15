import os
from functools import reduce
import time
import shutil
import csv
from torch.nn import Module

def get_cur_time_str():
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))


class CSVDataRecord:
    def __init__(self, file_path, header, backup=True):
        assert file_path.endswith('csv')

        self.file_path = file_path
        self.header = header
        
        if backup and os.path.exists(file_path):
            backup_file_path = '{}.{}'.format(file_path, get_cur_time_str())
            shutil.copyfile(file_path, backup_file_path)
            print('csv file already exists! backup raw file to {}'.format(backup_file_path))
    
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def write(self, data):
        assert len(data) == len(self.header)

        with open(self.file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)


exclude_dirs = [
    './legodnn/third_party'
]
include_exts = [
    '.py'
]


def get_is_dir_in_exclude_dirs(dir_abs_path):
    return reduce(lambda res, cur: res or dir_abs_path.startswith(os.path.abspath(cur)), exclude_dirs, False)


def get_is_filename_ends_with_include_exts(file_abs_path):
    return reduce(lambda res, cur: res or file_abs_path.endswith(cur), include_exts, False)


def count_file_source_code_line_num(file_abs_path):
    with open(file_abs_path, 'r') as f:
        lines = f.readlines()
        return len(lines)
    

def get_file_size(file_abs_path):
    return os.path.getsize(file_abs_path)


def recursive_count_source_code(dir_abs_path, data_record: CSVDataRecord):
    if get_is_dir_in_exclude_dirs(dir_abs_path):
        print('{} is in exclude_dirs, ignore all files in it'.format(dir_abs_path))
        return

    for p in os.listdir(dir_abs_path):
        p = os.path.join(dir_abs_path, p)
        
        if os.path.isfile(p) and get_is_filename_ends_with_include_exts(p):
            print('count file {}'.format(p))
            line_num = count_file_source_code_line_num(p)
            file_size = get_file_size(p)
            
            data_record.write([p, line_num, file_size])
            
        if os.path.isdir(p):
            print('enter dir {}'.format(p))
            recursive_count_source_code(p, data_record)


if __name__ == '__main__':
    recursive_count_source_code(os.path.abspath('./'), 
                                CSVDataRecord('./source-code-line-num.csv', ['file_path', 'line_num', 'file_size'], backup=False))
