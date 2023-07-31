import itertools
from utils.data_utils.txt_handler import TextHandler
import random
def synthetize(data):
    # synthetized_data = data
    synthetized_data = []
    col = [i for i in range(len(data))]
    row = [i for i in range(len(data))]
    comb_index = list(itertools.product(col,row))
    small_comb_index = random.sample(comb_index,100000)
    for index in small_comb_index:
        syn_token = data[index[0]]+ data[index[1]]
        synthetized_data.append(syn_token)
    return synthetized_data
if __name__ == '__main__':
    data = TextHandler('google-10000-english-usa-no-swears-medium.txt').read_txt()
    data = [t.replace('\n','') for t in data]
    synthetized_data = synthetize(data)
    TextHandler('..\synthetized_training_data.txt').write_txt(synthetized_data)