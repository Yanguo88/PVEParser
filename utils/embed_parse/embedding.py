import numpy as np
from collections import Counter
import os
from utils.embed_parse.process import prepare
import pandas as pd

chars = ['0','1','2','3','4','5','6','7','8','9',
             'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','other']
index = [i for i in range(len(chars))]

chars_dict = dict(zip(chars,index))

def word_embedding(word):
    assert  isinstance(word,str),'expected parameter type: string'
    word = word.lower()
    chars_count = Counter(word)
    embd_vector = [0]*len(chars)
    for char in word:
        char_num = chars_count.get(char)
        v_index = chars_dict.get(char,len(chars_dict)-1)
        embd_vector[v_index] = char_num/len(word)
    return embd_vector

def log_embedding(log_content):
    log_embd_vector = list() # shape:num_of_word,num_of_features
    for line in log_content:
        for word in line:
            log_embd_vector.append(word_embedding(word))
    return log_embd_vector

def store_embedding(file_path,save_name,word_column):
    '''
    :param file_path:
    :param word_column:
    :return:
    '''
    assert os.path.isfile(file_path),"file does not exist"
    file_path = os.path.abspath(file_path)
    data_dir = os.path.dirname(file_path)
    file_name = os.path.split(file_path)[1]

    embed_prepare = prepare(data_dir)

    file_content = embed_prepare.read_data(file_name,data_format='.xls')

    word_set = file_content[word_column].tolist()
    word_index = file_content.iloc[:,0].tolist()
    embedding_list = []
    for word in word_set:
        word_vector =list(map(str,word_embedding(word)))
        embedding_list.append(' '.join(word_vector))
    data = pd.DataFrame(zip(word_index,embedding_list),columns=['index','vector'])

    embed_prepare.save_data(save_name,data = data,data_format='.xls')
