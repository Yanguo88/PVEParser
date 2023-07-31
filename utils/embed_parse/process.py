
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import torch
import string
import math
from os import path
from scipy.special import perm,comb
from nltk import tokenize
import re

from collections import Counter
from itertools import chain

class prepare:
    '''
    用于读取日志数据和标签
    '''
    def __init__(self,data_dir):
        self.data_dir = data_dir
        pass


    def batch_iterator(self,X, y=None, batch_size=64,shuffle = False):
        """ Simple batch generator """
        n_samples = X.shape[0]
        indices = list(range(n_samples))
        if shuffle:
            random.seed(100)
            random.shuffle(indices)
        n_samples = X.shape[0]
        for i in np.arange(0, n_samples, batch_size):
            begin, end = i, min(i + batch_size, n_samples)
            batch_indices = indices[begin:end]
            if y is not None:
                yield X[batch_indices], y[batch_indices]
            else:
                yield X[batch_indices]

    def read_data(self, file_name, data_format='.txt'):
            data_path = os.path.join(self.data_dir,file_name)
            assert os.path.exists(data_path),'路径data_path 不存在！'
            data = None
            if data_format == '.txt' or data_format == '.log':
                with open(data_path,'r')as f:
                    data = f.readlines()
            elif data_format =='.npz':
                data = np.load(data_path)
            elif data_format =='.csv':
                data = pd.read_csv(data_path)
            elif data_format == '.xls' or data_format == '.xlsx':
                data = pd.read_excel(data_path)
            else:
                raise Exception('Invalid data format',data_format)
            return data

    def save_data(self, file_name, data, data_format='.txt'):
        '''
        :param file_name:
        :param seq_data:
        :param labels:
        :param data_format:
        :param separator: 当保存格式为文本是，一行数据包括了log sequence和lable,两者以separator相连
        :return:
        '''
        data_path = os.path.join(self.data_dir,file_name)
        #创建对应路径
        # if os.path.exists(data_path) is False:
        #     os.makedirs(os.path.dirname(data_path))
        if data_format == '.txt':
            with open(data_path,'w')as f:
                data =[str(line)+'\n' for line in data if not str(line).endswith('\n')]
                f.writelines(data)
        elif data_format == '.npz':
            #保存数据
            np.savez(data_path,**data)
        elif data_format == '.csv':
            data.to_csv(data_path,index=False)
        elif data_format == '.xls' or data_format == '.xlsx':
            writer = pd.ExcelWriter(data_path)
            data.to_excel(writer,sheet_name='Sheet1')
            writer.save()
            writer.close()
        else:
            raise Exception('Invalid data format', data_format)

    def get_item(self,file_name,return_label = True,data_format='.txt',separator='#'):
        '''
        设定日志数据和标签处于同一个文件中，获得数据和对应的标签
        :param file_name:经过sliding_window 截取后，保存的数据
        :param data_format:
        :param separator: 在txt格式的文本中，数据和标签以separator方式分割
        :return:
        '''
        x_data = []
        y_data = []

        if data_format == '.txt':
            txt_data = self.read_data(file_name, data_format)
            for line in txt_data:
                line = line.strip()
                line_sep=line.split(separator)
                x_data.append(line_sep[0])
                if return_label:
                    y_data.append(line_sep[1])
        elif data_format == '.npz':
            np_data = self.read_data(file_name, data_format)
            x_data = np_data['seq']
            if return_label:
                y_data = np_data['label']
        elif data_format == '.csv':
            pd_data = self.read_data(file_name, data_format)
            x_data = pd_data['seq']
            if return_label:
                y_data = pd_data['label']
        else:
            raise Exception('Invalid data format', data_format)
        return x_data,y_data

    def sliding_window(self,data,window_size,start_mark=None,end_mark=None):
        '''
        :param data:
        :param window_size: 滑动窗口大小
        :param start_mark: 开始符号，若不需要则不需赋值。类型为数值类型
        :param end_mark: 结束符号，若不需则不需赋值。类型为数值类型
        :return:
        '''
        sub_seqs = list()
        labels = list()
        for line in data:
            #统一减1，为了训练时的数据格式统一
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            line = self.insert_mark(line, start_mark, end_mark)
            #滑动窗口截取数据，以及获得响应的label
            for i in range(len(line)-window_size):
                sub_seq = line[i:i+window_size]
                label = line[i+window_size]
                sub_seqs.append(sub_seq)
                labels.append(label)
        return sub_seqs,labels

    def sliding_inverse(self,window_data,separator = ' '):
        '''
        由滑动窗口数据转化为原日志
        :param window_data: 滑动窗口数据，为二维数据,[1,len(window_data)]
        :param separator: 每一滑动窗口数据中，使用分隔符连接各个元素
        :return:
        '''
        parsed_logs = list()
        parsed_vectors = list()

        for index,line_window in enumerate(window_data):
            #第一个滑动窗口的数据全部记录
            log_vector = line_window[0].split(separator) if len(line_window)>0 else []
            log = line_window[0] if len(line_window)>0 else ''
            for index in range(1,len(line_window)):
                #其余的滑动窗口只记录最后一个
                new_word = line_window[index].split(separator)[-1]
                log = log + separator + new_word
                log_vector.append(new_word)
            parsed_vectors.append(log_vector)
            parsed_logs.append(log)
        return parsed_logs,parsed_vectors

    def insert_mark(self,line,start_mark,end_mark):
        '''
        插入开始和结束的标识符
        :param line:
        :param start_mark:
        :param end_mark:
        :return:
        '''
        if start_mark != None:
            line.insert(0, int(start_mark-1))
        if end_mark != None:
            line.append(int(end_mark-1))
        return line

    def gen_item(self,log_name,save_name,window_size = 3, data_format='.txt',separator='#',start_mark = None,end_mark=None):
        '''
        生成日志数据并保存
        :param log_name: 预处理后的数据文件名称
        :param data_format:
        :param separator: 生成的文本为txt时，x_data和y_data之间的分隔符
        :return:
        '''
        # 判断是否存在后缀名
        if len(os.path.splitext(save_name)) ==1:
            save_name = save_name+data_format
        log_path = os.path.join(self.data_dir, log_name)
        assert os.path.exists(log_path), '路径log_path 不存在！'
        log_data = self.read_data(log_name)
        seq_data,labels = self.sliding_window(log_data,window_size,start_mark=start_mark, end_mark=end_mark)
        if data_format == '.txt':
            txt_data = [' '.join(map(str,seq_data[i])) + separator + str(labels[i])for i in range(len(seq_data))]
            self.save_data(save_name, txt_data, data_format=data_format)
        elif data_format == '.npz':
            # 转换为numpy格式的数据
            txt_data = [' '.join(map(str,seq_data[i])) for i in range(len(seq_data))]
            np_data = np.array(txt_data, dtype=np.str)
            np_label = np.array(labels, dtype=np.str)
            #带有列名的numpy数据
            npz_data = {'seq':np_data,'label':np_label}
            # 保存数据
            self.save_data(save_name, npz_data, data_format=data_format)
        elif data_format == '.csv':
            csv_data = [[' '.join(map(str,seq_data[i])), labels[i]] for i in range(len(seq_data))]
            pd_data = pd.DataFrame(csv_data, columns=['seq', 'label'])
            self.save_data(save_name, pd_data, data_format=data_format)
        else:
            raise Exception('Invalid data format', data_format)
        print(save_name,'  训练文件已生成！')
        return seq_data,labels

    def model_data(self,mode,data_name,return_label,**other_args):
        '''
        生成或获得训练所需的数据
        :param mode: 'get' or 'gen'
        :param train_data_name: 训练文件的名称
        :param other_args: mode为gen时，传入生成训练数据是必要的参数，[log_name,window_size,gen_format,separator(optional),start_mark(optional),end_mark(optional),]
        :return:
        '''
        train_data_path = os.path.join(self.data_dir, data_name)

        train_data_format = os.path.splitext(train_data_path)[1]
        logs = []
        labels = []
        if mode == 'get':
            x_data,y_data = self.get_item(data_name,return_label = return_label ,data_format=train_data_format)
            for i, log in enumerate(x_data):
                # 日志异常检测中，这里需要转化为数字
                # log = list(map(int, log.strip().split()))
                logs.append(log)
                if return_label:
                    labels.append(int(y_data[i]))
        elif mode == 'gen':
            log_name = other_args.get('log_name')
            window_size = other_args.get('window_size')
            data_format = other_args.get('data_format')
            start_mark = other_args.get('start_mark',None)
            end_mark = other_args.get('end_mark',None)
            separator = other_args.get('separator','#')  #默认的分隔符为'#'
            logs,labels = self.gen_item(log_name,data_name,window_size=window_size,data_format=train_data_format,start_mark=start_mark,end_mark=end_mark,separator=separator)
        # logs = np.array(logs,dtype=int)
        # labels = np.array(labels,dtype= int)
        return logs,labels

    def down_sample(self,logs,labels,sample_ratio):
        print('sampling...')
        total_num = len(labels)
        all_index = list(range(total_num))
        sample_data = []
        sample_labels = []
        sample_num = int(total_num * sample_ratio)
        for i in tqdm(range(sample_num)):
            random_index = int(np.random.uniform(0, len(all_index)))
            sample_data.append(logs[random_index])
            sample_labels.append(labels[random_index])
            del all_index[random_index]
        return sample_data, sample_labels

    def training_data(self,name,return_label = True):
        '''
        :param name: 数据文件名称
        :return:
        '''
        logs, lables = self.model_data(mode='get',data_name=name,return_label = return_label)
        return logs,lables

    def test_data(self,name,mode = 'raw',return_label = True):
        '''
        :param name:
        :param mode: 模式为两种：raw 和  freq，前者是原始的窗口内容，后者是附加频率统计的结果,对应了两种不同的预测方式
        :return:
        '''

        X_data,y_data = self.model_data(mode='get',data_name=name,return_label = return_label)

        if mode == 'freq':
            dict_len = 0
            freq_dict = {}
            # 获取每个x对应的频率
            for i,line in enumerate(X_data):
                line.append(y_data[i])
                freq_dict[tuple(line)] = freq_dict.get(tuple(line),0)+1
                dict_len += 1
            return X_data,y_data,freq_dict
        elif mode == 'raw':
            return X_data,y_data

        # 在原版的代码里面，如果日志小于window_size,则以-1填充，但是这段代码仅限于在测试阶段使用，
        # 在训练阶段截取窗口数据时，忽略了长度小于window_size的日志
        # with open('../data/logdeep/' + name, 'r') as f:
        #     for ln in f.readlines():
        #         ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
        #         ln = ln + [-1] * (window_size + 1 - len(ln))
        #         hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
        #         length += 1
        # print('Number of sessions({}): {}'.format(name, len(data_dict)))

    def log_categorize(self,log_content,type_decision):
        #返回解析的日志，对应的日志模板和模板id
        '''
        :param parsed_log:解析过后的的排序向量
        :return:
        '''
        parsed_content = list()
        for line in log_content:
            parsed_line = []
            for word in line:
                if type_decision.get(str(word), 0) == 1:
                    parsed_line.append(str(word))
                elif len(str(word))==1 and (str(word) in string.punctuation):
                    pass
                # elif word =='':
                #     pass
                else:
                    parsed_line.append('<*>')
            parsed_content.append(parsed_line)
            # 1为模板词，0为参数词
            # parsed_content.append([str(word) if type_decision.get(str(word),0) == 1 else '*' for word in line])

        temp_ids = dict()
        log_ids = list()
        for index, parse_log in enumerate(parsed_content):
            str_log = ' '.join(parse_log)
            if str_log not in temp_ids.keys():
                temp_ids[str_log] = len(temp_ids) + 1
            parsed_content[index] = str_log
            log_ids.append(temp_ids[str_log])
        content = list(zip(log_ids, parsed_content))
        csv_data = pd.DataFrame(content, columns=['EventId', 'ParsedLog'])
        return csv_data

    def restore_log(self,log_vector,parsed_sorted_vector,stat_result = {}):
        #返回解析的日志，对应的日志模板和模板id

        template_logs = list()
        template_id_dict = dict()
        template_id_list = list()

        for index, parsed_line in enumerate(parsed_sorted_vector):
             line = parsed_line.strip().split()
             #最后一位是预测错误的标签,序列数据为line[0:-1]
             sorted_template = ' '.join(line[0:-1])
             end_key = ' '
             for key in stat_result.keys():
                 if sorted_template.endswith(key):
                    end_key = key
                    break
             if stat_result.get(end_key,True):
                 keywords = set(line[0:-1])
             else:
                 keywords = set(line)
             temp_line = list()
             for word in log_vector[index].strip().split():
                if word in keywords:
                    temp_line.append(word)
                else:
                    temp_line.append('*')
             template = ' '.join(temp_line)
             template_id = template_id_dict.get(template,len(template_id_dict)+1)
             template_id_dict[template] = template_id
             # print(template_id)
             template_id_list.append(template_id)
             template_logs.append(template)

        content = list(zip(template_id_list, template_logs, log_vector))

        csv_data = pd.DataFrame(content,columns = ['EventId','EventTemplate','Content'])

        return csv_data

class DatasetIterator(object):
    def __init__(self,dataset,batch_size,device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0
        self.device = device
        self.n_batches = math.floor(len(dataset)/batch_size)
        self.residue = False #记录batch数量是否为整数
        if len(dataset)%self.n_batches !=0:
            self.residue =True
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index*self.batch_size:len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index > self.n_batches:
            self.index =0
            raise StopIteration
        else:
            batches = self.dataset[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index +=1
            batches =self._to_tensor(batches)
            return batches
    def _to_tensor(self,datas):
        x = torch.LongTensor([item[0] for item in datas]).to(self.device) #样本数据 ids
        y = torch.LongTensor([item[1] for item in datas]).to(self.device) # 标签数据 label
        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device) #每一个序列的真实长度
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)    #
        return (x,seq_len,mask),y
    def __iter__(self):
        return self
    def __len__(self):
        if self.residue:
            return self.n_batches+1
        else:
            return self.n_batches

class BuildIterator(object):
    def __init__(self, X,y=None,batch_size = 64,shuffle = False):
        self.X = X # list
        self.y = y
        self.batch_size = batch_size
        self.index = 0
        self.n_batches = math.floor(len(X) / batch_size)
        self.residue = False  # 记录batch数量是否为整数

        if len(X) % self.n_batches != 0:
            self.residue = True
        self.indices = list(range(len(X)))


        if shuffle:
            random.seed(100)
            random.shuffle(self.indices)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batch_indices = self.indices[self.index * self.batch_size:len(self.X)]
            batche_X = self.X[batch_indices]
            self.index += 1
            if self.y is not None:
                batche_y = self.y[batch_indices]
                return  batche_X,batche_y
            else:
                return batche_X
        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:

            batch_indices = self.indices[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            batche_X = self.X[batch_indices]
            self.index += 1
            if self.y is not None:
                batche_y = self.y[batch_indices]
                return batche_X, batche_y
            else:
                return batche_X

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

class preprocess:
    def __init__(self,data_dir,file_names,split_again = False):
        self.data_dir = data_dir
        self.files = file_names
        self.word_file = 'words.txt'
        self.char_file = 'chars.txt'
        self.split_again = split_again
        self.regex_list = {
            # 'HDFS': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
            # 'HDFS':[],
            'Hadoop': [r'(\d+\.){3}\d+'],
            'Spark': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
            'Zookeeper': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
            'BGL': [r'core\.\d+'],
            'HPC': [r'=\d+'],
            'Thunderbird': [r'(\d+\.){3}\d+'],
            'Windows': [r'0x.*?\s'],
            'Mac': [r'([\w-]+\.){2,}[\w-]+'],
            'OpenStack': [r'([ |: | \( | \) | "|\{|\}|@|$|\[|\]|\||;])'],
            # 'Linux': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],

            'Andriod': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+',
                        r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],

            # 'HealthApp': [],
            # 'Apache': [r'(\d+\.){3}\d+'],
            # 'OpenSSH': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
            # 'Proxifier': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
            # 'OpenStack': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],

        }

        # self.paths = [path.join(data_dir,file_name) for file_name in files ]
    def read_data(self, file_name, data_format='.txt'):
        data_path = os.path.join(self.data_dir, file_name)
        assert os.path.exists(data_path), 'data_path 不存在！'
        data = None
        if data_format == '.txt' or data_format == '.log':
            with open(data_path, 'r')as f:
                data = f.readlines()
        elif data_format == '.npz':
            data = np.load(data_path)
        elif data_format == '.csv':
            data = pd.read_csv(data_path, engine='python' )
        elif data_format == '.xls' or data_format == '.xlsx':
            data = pd.read_excel(data_path)
        else:
            raise Exception('Invalid data format', data_format)
        return data

    def text_preprocess(self,regex_name,line):
            for currentRex in self.regex_list[regex_name]:
                line = re.sub(currentRex, ' ', line)
            return line

    def save_data(self, file_name, data, data_format='.txt'):
        '''
        :param file_name:
        :param seq_data:
        :param labels:
        :param data_format:
        :param separator: 当保存格式为文本是，一行数据包括了log sequence和lable,两者以separator相连
        :return:
        '''
        data_path = os.path.join(self.data_dir, file_name)
        # 创建对应路径
        # if os.path.exists(data_path) is False:
        #     os.makedirs(os.path.dirname(data_path))
        if data_format == '.txt':
            with open(data_path, 'w')as f:
                data = [line + '\n' for line in data if not line.endswith('\n')]
                f.writelines(data)
        elif data_format == '.npz':
            # 保存数据
            np.savez(data_path, **data)
        elif data_format == '.csv':
            data.to_csv(data_path, index=False)
        elif data_format == '.xls' or data_format == '.xlsx':
            writer = pd.ExcelWriter(data_path)
            data.to_excel(writer, sheet_name='Sheet1')
            writer.save()
            writer.close()
        else:
            raise Exception('Invalid data format', data_format)
    def gen_corpus(self,use_column):
        chars = []
        words = []
        tk = tokenize.WhitespaceTokenizer()
        chars_path = path.join(self.data_dir, self.char_file)
        # 检查文件是否存在，并根据split_again 生成字符级的语料库
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
                    line = list(map(str.lower, tk.tokenize(line)))
                    # line = self.text_preproc(line)
                    words.append(line)
                    # words.append(map(str.lower,tokenize.word_tokenize(line)))
            for line_words in words:
                for word in line_words:
                    chars.append([char for char in word])
            txt_words = [' '.join(line)for line in words]
            txt_chars = [' '.join(line_char) for line_char in chars]
            self.save_data(file_name='words.txt',data= txt_words)
            self.save_data(file_name='chars.txt',data= txt_chars)
        return chars
    def log_tokenize(self,use_column):
        log_contents = dict()
        tk = tokenize.WhitespaceTokenizer()

        for file_name in self.files:
            csv_content = self.read_data(file_name, data_format='.csv')

            log_contents[file_name] = csv_content[use_column].tolist()
        for file_name in log_contents.keys():
            log_content = log_contents[file_name]

            regex_name = [key for key in self.regex_list.keys() if key in file_name]
            for index,line in enumerate(log_content):
                if len(regex_name):
                    line = self.text_preprocess(regex_name[0], line)

                line = list(map(str.lower,tk.tokenize(line)))
                line = list(map(str.lower, self.regex_replace(line)))
                log_contents[file_name][index] = line

                # log_contents[file_name][index] = list(map(str.lower, self.regex_replace(line)))
        return log_contents
    def word_freq(self, column_name):
        log_contents = []
        words = []
        tk = tokenize.WhitespaceTokenizer()
        # tk = tokenize.RegexpTokenizer(pattern=r"\s|=|:", gaps=True)
        for file_name in self.files:
            content = self.read_data(file_name, data_format='.csv')

            regex_name = [key for key in self.regex_list.keys() if key in file_name]
            for line in content[column_name]:
                if len(regex_name):
                    line = self.text_preprocess(regex_name[0],line)
                line = list(map(str.lower, tk.tokenize(line)))
                line = list(map(str.lower, self.regex_replace(line)))
                words.append(line)
        #     log_contents.append(content[column_name])
        #
        # for content in log_contents:
        #     content = self.regex_replace(content)
        #     for line in content:
        #         line = list(map(str.lower,tk.tokenize(line)))
        #         words.append(line)
        words = list(chain.from_iterable(words))
        words_count = Counter(words)
        sorted_freq = sorted((words_count.items()), key=lambda x: x[1], reverse=True)
        pd_data = pd.DataFrame(sorted_freq, columns=['word', 'frequency'])
        self.save_data('word_frequency.xls',pd_data,data_format='.xls')
    def annlysis(self,result_files,use_columns):
        for file_name in result_files:
            pd_data = self.read_data(file_name,data_format='.xls')
            use_data = pd_data[use_columns]
            keys = set(use_data[use_columns[0]].tolist())
            print(file_name,":")
            for key in keys:
                row_indices = use_data[use_columns[0]].isin([key])
                values = set(use_data[use_columns[1]][row_indices].tolist())
                print(key,' '.join(list(map(str,values))))
    def word_filter(self, word_file, column_name):
        # 排除带有纯数字和字符的token
        word_freq = self.read_data(word_file,data_format='.xls')
        word_set = word_freq[column_name]
        exclude_chars = string.punctuation+string.digits
        # trantab = str.maketrans(exclude_chars,''.join([' 'for _ in exclude_chars]))
        exclude_flags = []
        for word in word_set:
            is_include = False
            for char in str(word):
                if char not in exclude_chars:
                    is_include = True
                    exclude_flags.append(is_include)
                    break
            if not is_include:
                exclude_flags.append(is_include)
        filter_result =  word_set[exclude_flags].tolist()

        pd_data = pd.DataFrame(zip(filter_result),columns=['word'])
        self.save_data('word_filter.xls',data=pd_data,data_format='.xls')


    def regex_replace(self,line):

        line = [re.sub(re.compile(r'/\w+(/.+)+', re.I), " ", w) for w in line]
        # line = [re.sub(re.compile(r'\w*\d\w*', re.I), " ", w) for w in line]
        #
        # line = [re.sub(r'(:(?=\s))|((?<=\s):)', " ", w) for w in line]
        # line = [re.sub(r'(\d+\.)+\d+', " ", w) for w in line]
        # line = [re.sub(r'\d{2}:\d{2}:\d{2}', " ", w) for w in line]
        # line = [re.sub(r'Mar|Apr|Dec|Jan|Feb|Nov|Oct|May|Jun|Jul|Aug|Sep', " ", w) for w in line]
        # line = [re.sub(r':?(\w+:)+', "", w) for w in line]
        #
        # #过滤符号
        # line = [re.sub(r'\.|\(|\)|\<|\>|\/|\-|\=|\[|\]|,|:', " ", w) for w in line]
        # line = [re.sub(r'\#', " ", w) for w in line]
        # line = [re.sub(r'\|', " ", w) for w in line]
        # line = [re.sub(r'\"', " ", w) for w in line]
        # line = [re.sub(r'\;|\@', " ", w) for w in line]
        #
        # line = [re.sub(r'\\b(0[xX])?[A-Fa-f0-9]+\\b', " ", w) for w in line]  ##过滤十六进制的内存地址
        # line = [re.sub(r'\s?(\s|^)[1-9]\d*(\s|$)\s?', " ", w) for w in line]  ###过滤连续的纯数字
        # line = [re.sub(r'0[xX]\w+', " ", w) for w in line]  ##过滤十六进制数

        # line = [re.sub(r'((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}', ' ', w) for w
        #         in line]  ##过滤ip地址
        # line = [re.sub(r'[a-zA-z]+://[^\s]*', ' ', w) for w in line] # 过滤地址
        # line = [re.sub(r'\\b(0[xX])?[A-Fa-f0-9]+\\b', ' ', w) for w in line]  #过滤十六进制的内存地址
        # line = [re.sub(r'0[xX]\w+', ' ', w) for w in line]  ##过滤十六进制数
        #
        # # line = [re.sub(r'Mar|Apr|Dec|Jan|Feb|Nov|Oct|May|Jun|Jul|Aug|Sep',' ', w) for w in line]
        # # line = [re.sub(r'[a-zA-z\d.]+:[\d]+', ' ', w) for w in line]
        # line = [re.sub(r'[(|\[](.*?)[)\]]',' ', w) for w in line]
        # line = [re.sub(r'=', ' ', w) for w in line]
        return line

if __name__ == '__main__':

    # data_prepare = prepare(data_dir='../../data/rnn_parse')
    # other_args = {
    #     'log_name': 'sorted_vector_hdfs.log',
    #     'window_size': 3,
    #     'data_format': '.csv',
    #     'separator': '#',
    #     'start_mark': None,
    #     'end_mark': 2308
    # }
    # data_prepare.model_data(mode='gen',data_name='training_data_hdfs.csv',**other_args)
    # data_prepare.training_data(mode='gen', train_data_name='training_data_hdfs.npz')
    # data_prepare.test_data('training_data_hdfs.npz')
    data_dir = '../../data/embed_parse'
    save_dir = '../../results/embed_parse'
    file_names = \
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

    data_prep = preprocess(data_dir, file_names, split_again=True)
    # data_prep.gen_corpus(use_column='Content')
    data_prep.word_freq(column_name='Content')
    data_prep.word_filter('word_frequency.xls','word')

    # 分析解析结果和ground_truth中EventId 对应结果
    # data_prep = preprocess(save_dir, file_names, split_again=True)
    # result_files =[file_name.split('_')[0]+'.xls' for file_name in file_names]
    # data_prep.annlysis(result_files,['EventId','ParsedId'])

    str_conj = 'processHandleBroadcastAction'
    # tk = tokenize.word_tokenize()
    print(tokenize.word_tokenize(str_conj))
    # print(re.findall(r'[A-Za-z0-9.\\]+$','l'))