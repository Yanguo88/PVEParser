import os
import numpy as np
import  pandas as pd
from os import path
import nltk
from nltk import tokenize
class data_process:
    def __init__(self,data_dir,file_names,split_again = False):
        self.data_dir = data_dir
        self.files = file_names
        self.word_file = 'words.txt'
        self.char_file = 'chars.txt'
        self.split_again = split_again
        # self.paths = [path.join(data_dir,file_name) for file_name in files ]

    def read_data(self, file_name, data_format='.txt'):
        data_path = os.path.join(self.data_dir, file_name)
        assert os.path.exists(data_path), 'data_path does not exist'
        data = None
        if data_format == '.txt' or data_format == '.log':
            with open(data_path, 'r')as f:
                data = f.readlines()
        elif data_format == '.npz':
            data = np.load(data_path)
        elif data_format == '.csv':
            data = pd.read_csv(data_path)
        elif data_format == '.xls' or data_format == '.xlsx':
            data = pd.read_excel(data_path)
        else:
            raise Exception('Invalid data format', data_format)
        return data
    def save_data(self, file_name, data, data_format='.txt'):
        '''
        :param file_name:
        :param seq_data:
        :param labels:
        :param data_format:
        :return:
        '''
        data_path = os.path.join(self.data_dir, file_name)

        if data_format == '.txt':
            with open(data_path, 'w')as f:
                data = [line + '\n' for line in data if not line.endswith('\n')]
                f.writelines(data)
        elif data_format == '.npz':

            np.savez(data_path, **data)
        elif data_format == '.csv':
            data.to_csv(data_path, index=False)
        elif data_format == '.xls' or data_format == '.xlsx':
            writer = pd.ExcelWriter(data_path)
            data.to_excel(writer, sheet_name='word_embedding')
            writer.save()
            writer.close()
        else:
            raise Exception('Invalid data format', data_format)

    def gen_corpus(self,use_column):
        chars = []
        words = []
        chars_path = path.join(self.data_dir, self.char_file)

        if path.exists(chars_path) and not self.split_again:
            txt_chars = self.read_data(self.char_file)
            for char_line in txt_chars:
                chars.append(char_line.split())
        else:
            log_contents= []
            for file_name in self.files:
                content = self.read_data(file_name,data_format='.csv')
                log_contents.append(content[use_column])
            for content in log_contents:
                for line in content:
                    words.append(tokenize.wordpunct_tokenize(line))

            for line_words in words:
                for word in line_words:
                    chars.append([char for char in word])

            txt_words = [' '.join(line)for line in words]
            txt_chars = [' '.join(line_char) for line_char in chars]
            self.save_data(file_name='words.txt',data= txt_words)
            self.save_data(file_name='chars.txt',data= txt_chars)
        return chars

if __name__ == '__main__':
    data_dir = '../../data/embed_parse'
    file_names =\
        [
            'Andriod_2k.log_structured.csv',
            'Apache_2k.log_structured.csv',
            'BGL_2k.log_structured.csv',
            'Hadoop_2k.log_structured.csv',
            'HDFS_2k.log_structured.csv',
            'HealthApp_2k.log_structured.csv',
            'HPC_2k.log_structured.csv',
            'Linux_2k.log_structured.csv',
            'Mac_2k.log_structured.csv',
            'OpenSSH_2k.log_structured.csv',
            'OpenStack_2k.log_structured.csv',
            'Proxifier_2k.log_structured.csv',
            'Spark_2k.log_structured.csv',
            'Thunderbird_2k.log_structured.csv',
            'Windows_2k.log_structured.csv',
            'Zookeeper_2k.log_structured.csv'
        ]
    data_prep = data_process(data_dir, file_names, split_again=True)
    data_prep.gen_corpus(use_column='Content')