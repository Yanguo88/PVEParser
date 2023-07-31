#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

sys.path.append('../')
import torch

from models.embedparse import embedding_parse
from tools.embedparse import trainer,predicter
from utils.embed_parse.utils import seed_everything

# Config Parameters
options = dict()
options['data_dir'] = '../data/embed_parse/'
options['device'] = 'cpu'
options['save_dir'] = "../results/embed_parse/"

# Model paremeter
options['char_dim'] = 10
options['word_dim'] = 10
options['latent_dim'] = 100
options['hidden_dim'] = 20
options['padding_size'] = 15

# options['training_data'] = 'englishword/google-10000-english-usa-no-swears-medium.txt'
# options['validation_data'] = 'englishword/google-10000-english-usa-no-swears-medium.txt'
# options['test_data'] = 'englishword/google-10000-english-usa-no-swears-medium.txt'
options['training_data'] = 'synthetized_training_data.txt'
options['validation_data'] = 'synthetized_training_data.txt'
options['test_data'] = 'synthetized_training_data.txt'

# Train
options['batch_size'] = 512
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.002
options['max_epoch'] = 370
options['lr_step'] = (300,350)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "embedparse"

# Predict
options['model_path'] = "../results/embed_parse/embedparse_bestloss.pth"
options['loss_threshold'] = 0.001
options['ground_truth'] = [
    'HDFS_2k.log_structured.csv',
    'Hadoop_2k.log_structured.csv',
    'Spark_2k.log_structured.csv',
    'Zookeeper_2k.log_structured.csv',
    'OpenStack_2k.log_structured.csv',
    'BGL_2k.log_structured.csv',
    'HPC_2k.log_structured.csv',
    'Thunderbird_2k.log_structured.csv',
    'Windows_2k.log_structured.csv',
    'Linux_2k.log_structured.csv',
    'Mac_2k.log_structured.csv',
    'Andriod_2k.log_structured.csv',
    'HealthApp_2k.log_structured.csv',
    'Apache_2k.log_structured.csv',
    'OpenSSH_2k.log_structured.csv',
    'Proxifier_2k.log_structured.csv',
]

# Config Parameters
seed_everything(seed=1234)


def train():
    model = embedding_parse(options['save_dir'],options['char_dim'],options['word_dim'],
                            options['latent_dim'],options['hidden_dim'],options['padding_size'],options['device'])
    train = trainer(model, options)
    train.start_train()

def predict():
    model = embedding_parse (options['save_dir'],options['char_dim'],options['word_dim'],
                            options['latent_dim'],options['hidden_dim'],options['padding_size'],options['device'])
    predict = predicter(model, options)
    # predict.predict_loss()
    predict.predict_loss_with_similarity()
    predict.evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', choices=['train', 'predict'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        predict()
