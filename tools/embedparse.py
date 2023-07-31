import gc
import os
import time
import pandas as pd
import torch
import random
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import re
from utils.embed_parse.process import prepare
from utils.embed_parse.process import BuildIterator
from utils.embed_parse import evaluator
from utils.embed_parse.process import preprocess


class trainer:
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        self.data_dir = options['data_dir']
        self.batch_size = options['batch_size']
        self.device = options['device']
        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']
        self.max_epoch = options['max_epoch']
        self.training_data = options['training_data']
        self.validation_data = options['validation_data']
        self.use_label = False
        self.data_prepare = prepare(data_dir=self.data_dir)
        self.train_loader, self.valid_loader, self.num_train_log, self.num_valid_log = self.build_loader ()
        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        # self.train_model = rnn_model.to(self.device)
        self.model = model.to(self.device)
        self.optimizer = self.init_optimizer(options['optimizer'], options['lr'])
        self.criterion = self.model.loss_function
        if options['resume_path'] is not None:
            self.resume(options['resume_path'])
        self.save_parameters(options, self.save_dir + "parameters.txt")

        self.record = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }

    def valid_step(self):
        self.model.eval()
        total_losses = 0
        # tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)
        for i, word_x in enumerate(self.valid_loader):
            with torch.no_grad():

                result = self.model(x=word_x)
                loss,recons_loss= self.criterion(*result, kld_weight= 0.5)
                total_losses += float(loss)
        return total_losses

    def train_step(self):
        self.model.train()
        # tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        total_losses = 0
        for i, word_x in enumerate(self.train_loader):
            result = self.model(x = word_x)
            # result 的内容为 [self.decode(z), input, mu, log_var]
            loss,recon_loss = self.criterion(*result,kld_weight = 0.5)
            total_losses += float(loss)
            total_recon_losses = float(recon_loss)
            loss /= self.accumulation_step
            loss.backward()
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            # tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))
        return total_losses

    def get_data(self):
        train_logs,train_labels = self.data_prepare.training_data(name=self.training_data,return_label=self.use_label)
        train_logs = np.array(train_logs,dtype=str)

        val_logs, val_labels = self.data_prepare.training_data(name=self.validation_data,return_label=self.use_label)
        val_logs = np.array(val_logs, dtype=str)

        num_train_log = len(train_logs)
        num_valid_log = len(val_logs)
        return train_logs, val_logs, num_train_log, num_valid_log

    def build_loader(self):

        train_logs, train_labels = self.data_prepare.training_data(name=self.training_data, return_label=self.use_label)
        train_logs = np.array(train_logs, dtype=str)

        val_logs, val_labels = self.data_prepare.training_data(name=self.validation_data, return_label=self.use_label)
        val_logs = np.array(val_logs, dtype=str)

        if self.use_label:
            train_loader = BuildIterator(X=train_logs,y=train_labels,batch_size=self.batch_size,shuffle= False)
            val_loader = BuildIterator(val_logs,val_labels, batch_size=self.batch_size,shuffle= False)
        else:
            train_loader = BuildIterator(train_logs,batch_size= self.batch_size)
            val_loader = BuildIterator(val_logs, batch_size=self.batch_size,shuffle=True)
        num_train_log = len(train_logs)
        num_valid_log = len(val_logs)
        return train_loader, val_loader, num_train_log, num_valid_log

    def remove_space(self, *params):
        for param in params:
            del param
        gc.collect()

    def init_optimizer(self, opti_name, learning_rate):
        if opti_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9)
        elif opti_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError
        return optimizer

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.record = checkpoint['record']
        self.best_score = checkpoint['best_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def add_record(self, outer_key, inner_dict):
        for inner_key, value in inner_dict.items():
            self.record[outer_key][inner_key].append(value)

    def save_parameters(self, options, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w+") as f:
            for key in options.keys():
                f.write("{}: {}\n".format(key, options[key]))

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "record": self.record,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = self.save_dir + self.model_name + "_" + suffix + ".pth"
        torch.save(checkpoint, save_path)
        # print("Save rnn_model checkpoint at {}".format(save_path))

    def start_train(self):
        inner_dict = dict()
        msg = 'Starting epoch: {0:<5d}| phase:{1:<10}|⏰: {2:<10}| Learning rate: {3:.5f} | {1}loss: {4}'
        last_improve_epoch = self.start_epoch

        break_flag = False
        for epoch in range(self.start_epoch, self.max_epoch):
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if epoch in self.lr_step:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio

            #train
            inner_dict['time'] = time.strftime("%H:%M:%S")
            inner_dict['loss'] = self.train_step()
            inner_dict['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            inner_dict['epoch'] = epoch
            print(msg.format(inner_dict['epoch'], 'train', inner_dict['time'], inner_dict['lr'],inner_dict['loss']))

            self.add_record(outer_key='train', inner_dict=inner_dict)

            if epoch % 2 == 0:
                inner_dict['loss'] = self.valid_step()
                inner_dict['time'] = time.strftime("%H:%M:%S")
                if (inner_dict['loss'] < self.best_loss):
                    print(msg.format(inner_dict['epoch'], 'valid', inner_dict['time'], inner_dict['lr'],inner_dict['loss'])+'(best so far)\n')
                else:
                    print(msg.format(inner_dict['epoch'], 'valid', inner_dict['time'], inner_dict['lr'],inner_dict['loss'])+'\n')

                self.add_record(outer_key='train', inner_dict=inner_dict)
                self.save_checkpoint(epoch,save_optimizer=False,suffix="epoch" + str(epoch))
                if (epoch-last_improve_epoch< 100):
                    if (inner_dict['loss'] < self.best_loss):
                        last_improve_epoch = epoch
                        self.best_loss = inner_dict['loss'] if (inner_dict['loss'] < self.best_loss) else self.best_loss
                        self.save_checkpoint(epoch,save_optimizer=True,suffix="bestloss")
                else:
                    break_flag = True

            if break_flag:
                self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
                break
            elif epoch == self.max_epoch-1:
                self.save_checkpoint(epoch, save_optimizer=True, suffix="last")

    def save_record(self):
        try:
            for key, values in self.record.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv",
                                            index=False)
            print("Record saved")
        except:
            print("Failed to save record")


class predicter:
    def __init__(self, model, options):
        self.data_dir = options['data_dir']
        self.save_dir = options['save_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']

        self.loss_threshold = options['loss_threshold']


        self.data_prepare = prepare(data_dir=self.data_dir)
        self.test_data = options['test_data']

        self.ground_truths = options['ground_truth']

        self.sliding_line = list()
        self.break_flag = False

        self.sliding_datas = list()
        self.criterion = self.model.loss_function

    def evaluation(self,ground_truth = ''):
        pd_dec = self.data_prepare.read_data('word_decisions1.xls',data_format='.xls')
        dict_dec = dict(zip(pd_dec['word'].tolist(),pd_dec['decision'].tolist()))
        local_ground_truths = self.ground_truths if ground_truth == '' else [ground_truth]
        log_prep = preprocess(self.data_dir,local_ground_truths)
        logs_content = log_prep.log_tokenize(use_column='Content')
        parsed_content = dict()
        for key in logs_content.keys():
            single_content = logs_content[key]
            parsed_data = self.data_prepare.log_categorize(single_content, dict_dec)
            parsed_content[key] = parsed_data

        for ground_truth in local_ground_truths:
            ground_truth_data = self.data_prepare.read_data(ground_truth, data_format='.csv')
            # d = parsed_content[ground_truth]['EventId']
            precision, recall, f_measure, accuracy, rand_index = evaluator.get_metrics(
            groundtruth=ground_truth_data['EventId'], parsed_ids=parsed_content[ground_truth]['EventId'])

            print('{:15} Precision:{:.4f} , Recall: {:.4f}, F1_measure: {:.4f}, '
                  'Randindex: {:.4f},Parsing_Accuracy: {:.4f}'.format(
                ground_truth.split('_')[0], precision, recall, f_measure,rand_index, accuracy ))

            data = zip(ground_truth_data['EventId'].tolist(),parsed_content[ground_truth]['EventId'],
                ground_truth_data['Content'],parsed_content[ground_truth]['ParsedLog'],ground_truth_data['EventTemplate']
                )
            pd_data=pd.DataFrame(data,columns=['EventId','ParsedId','Log','EventTemplate','TruthTemplate'])
            # accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean, edit_distance_result_std = \
            #     evaluator.evaluate(ground_truth_data, pd_data)
            log_prep.data_dir = self.save_dir
            log_prep.save_data(ground_truth.split('_')[0]+'.xls',pd_data,data_format='.xls')
        pass
    def predict_loss(self):
        model = self.model.to(self.device)

        word_freq = self.data_prepare.read_data(file_name='word_frequency.xls',data_format='.xls')

        num_error = 0

        words_type = dict()

        model.load_state_dict(torch.load(self.model_path,map_location=torch.device(self.device))['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        # Test the rnn_model
        start_time = time.time()
        word_set = word_freq['word']
        all_loss = []
        with torch.no_grad():

            for word in tqdm(word_set):
                result = self.model(x = [str(word)])
                loss,recons_loss = self.criterion(*result, kld_weight= 0.5)

                index = word_freq[word_freq.word == word].index.tolist()
                freq = word_freq.loc[index, 'frequency'].tolist()
                freq = freq[0] if len(freq) > 0 else 0
                all_loss.extend([recons_loss.tolist()])

                words_type[str(word)] = 1 if self.loss_threshold >= recons_loss else 0
                if self.hasNumbers(str(word)):
                    words_type[str(word)] = 0

        pd_dec = pd.DataFrame(data=zip(words_type.keys(),words_type.values()), columns=['word', 'decision'])
        self.data_prepare.save_data('word_decisions1.xls', pd_dec, data_format='.xls')
        self.data_prepare.save_data('all_loss.txt', all_loss)

        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))
        return words_type
    def predict_loss_with_similarity(self):
        self.model = self.model.to(self.device)

        word_freq = self.data_prepare.read_data(file_name='word_frequency.xls',data_format='.xls')

        words_type = dict()

        self.model.load_state_dict(torch.load(self.model_path,map_location=torch.device(self.device))['state_dict'])
        self.model.eval()
        print('model_path: {}'.format(self.model_path))
        # Test the rnn_model
        start_time = time.time()

        word_set = [ str(word).strip() for word in word_freq['word'] if len(str(word).strip())>=1]
        all_loss = []
        test_sample = self.model.sample(1000,self.device)
        with torch.no_grad():

            for word in tqdm(word_set):
                word_embedding = self.model.integrate_embedding1(words = [str(word)])
                word_embedding = word_embedding.repeat(1000,1)
                recons_loss = F.mse_loss(word_embedding,test_sample)

                all_loss.extend([recons_loss.tolist()])
                # 1 template token， 0 parameter token
                words_type[str(word)] = 1 if self.loss_threshold >= recons_loss else 0
                if self.hasNumbers(str(word)):
                    words_type[str(word)] = 0

        pd_dec = pd.DataFrame(data=zip(words_type.keys(),words_type.values()), columns=['word', 'decision'])
        self.data_prepare.save_data('word_decisions1.xls', pd_dec, data_format='.xls')
        self.data_prepare.save_data('all_loss.txt', all_loss)

        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))
        return words_type
    def predict_loss_with_single_log(self,file_name):
        model = self.model.to(self.device)

        model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device))['state_dict'])
        model.eval()

        csv_content = self.data_prepare.read_data(file_name=file_name, data_format='.csv')
        log_prep = preprocess(self.data_dir, [file_name])
        log_prep.word_freq(column_name='Content')
        word_freq = self.data_prepare.read_data(file_name='word_frequency.xls', data_format='.xls')
        print('model_path: {}'.format(self.model_path))
        # Test the rnn_model
        start_time = time.time()
        word_set = word_freq['word']
        all_loss = []

        words_type = dict()
        with torch.no_grad():

            for word in tqdm(word_set):
                result = self.model(x=[str(word)])
                loss, recons_loss = self.criterion(*result, kld_weight=0.5)
                all_loss.extend([recons_loss.tolist()])
                words_type[str(word)] = 1 if self.loss_threshold >= recons_loss else 0
                if self.hasNumbers(str(word)):
                    words_type[str(word)] = 0

        pd_dec = pd.DataFrame(data=zip(words_type.keys(), words_type.values()), columns=['word', 'decision'])
        self.data_prepare.save_data('word_decisions1.xls', pd_dec, data_format='.xls')
        self.data_prepare.save_data('all_loss.txt', all_loss)
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))
        return words_type

    def hasNumbers(self, s):
        pattern = re.compile(r'(0[xX])?[A-Fa-f0-9]{5,}')
        return any(char.isdigit() for char in s) or (pattern.search(s) is not None)
