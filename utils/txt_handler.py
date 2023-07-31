import pickle
import os
class TextHandler:
    def __init__(self, file):
        self.file = file

    def read_txt(self, encoding='utf-8'):
        """读取pickle数据"""
        with open(self.file, 'r') as f_read:
             data = f_read.readlines()
        return data
    def read_txt_chunk(self,chunk_size = 500*1024*1024):
        """
         Lazy function (generator) to read a file piece by piece.
         Default chunk size: 1M
         You can set your own chunk size
         """
        file_object = open(self.filePath)
        while True:
            chunk_data = file_object.read(chunk_size)
            if not chunk_data:
                break
            yield chunk_data

    def write_txt(self, data, encoding='utf-8'):
        """向pickle文件写入数据"""

        if not os.path.exists(self.file):
            file_dir = os.path.dirname(self.file)
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)

        with open(self.file, 'w')as f_save:
            f_save.writelines([line+'\n' for line in data])

    def write_txt_append(self,data,encoding='utf-8'):
        if not os.path.exists(self.file):
            file_dir = os.path.dirname(self.file)
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)
        with open(self.file, 'a')as f_save:
            f_save.writelines([line + '\n' for line in data])



if __name__ == '__main__':
    TextHandler('../../models/parameters/rawlog_params.txt').write_txt(['3','4'])