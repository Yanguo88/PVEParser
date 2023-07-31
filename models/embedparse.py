import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import word2vec
import os
from itertools import chain
import copy
import random
from os import path
import numpy as np
class embedding_parse(nn.Module):
    def __init__(self,save_dir,char_dim,word_dim,
                 latent_dim,hidden_dim,padding_size,device):
        super(embedding_parse, self).__init__()
        self.save_dir = save_dir
        self.char_dim = char_dim
        self.word_dim = word_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.padding_size = padding_size
        self.embed_cdict = dict()
        self.embed_wdict = dict()
        self.ngram_model = charcnn(char_dim=self.char_dim)
        self.embed_model = charembed(sg=0, window=3, embed_dim=char_dim, save_dir= self.save_dir, iter=100)
        # self.word_embed = nn.Linear(self.char_dim*self.padding_size,self.word_dim)
        self.word_embed = nn.Linear(self.char_dim, self.word_dim)
        self.vae = VAE(input_dim= self.char_dim,latent_dim= self.latent_dim,hidden_dim= self.hidden_dim)
        self.loss_function = self.vae.loss_function
        self.sample = self.vae.sample
    def forward(self,x):
       x = self.integrate_embedding1(x)
       return self.vae(x)

    def integrate_embedding_test(self,words):
        words_embedding,chars_embedding = self.words_embedding(words)
        chars_embedding = chars_embedding.unsqueeze(1)
        ngram_embedding =self.ngram_model(chars_embedding)
        ngram_embedding = ngram_embedding.squeeze(-1)
        integrated_embeddings = torch.cat([words_embedding,ngram_embedding],dim=1)
        return integrated_embeddings,words_embedding,ngram_embedding

    def integrate_embedding1(self,words):

        words_embedding,chars_embedding = self.words_embedding_v1(words)
        chars_embedding = chars_embedding.unsqueeze(1)
        ngram_embedding = self.ngram_model(chars_embedding)
        ngram_embedding = ngram_embedding.squeeze(-1)
        integrated_embeddings = torch.cat([words_embedding,ngram_embedding],dim=1)
        return ngram_embedding

    # def integrate_embedding1(self, words):
    #     words_embedding, chars_embedding = self.words_embedding_v1(words)
    #     chars_embedding = chars_embedding.unsqueeze(1)
    #
    #     integrated_embeddings = chars_embedding
    #     return words_embedding

    def words_embedding(self,words):
        self.embed_model.get_model()
        # shape = [1,batch_size,padding_size*char_dim]
        words_embedding = []
        #shape = [1,batch_size,padding_size,char_dim]
        chars_embedding = []
        empty_embedding = [0.0]*self.char_dim
        for word in words:
            if word in self.embed_wdict.keys():
               word_embedding = self.embed_wdict[word]
            else:
               word_embedding = self.embed_model.word_embedding(word)
               self.embed_wdict[word] = word_embedding
            words_embedding.append(word_embedding)
        for index,word_embed in enumerate(words_embedding):
            if len(word_embed) >= self.padding_size:
                words_embedding[index] = word_embed[:self.padding_size]
            else:
                gap_size = self.padding_size - len(word_embed)
                for _ in range(gap_size): words_embedding[index].append(empty_embedding)
            chars_embedding.append(copy.deepcopy(words_embedding[index]))
            words_embedding[index] = list(chain.from_iterable(words_embedding[index]))
        words_embedding = torch.tensor(words_embedding).to(self.device)
        chars_embedding = torch.tensor(chars_embedding).to(self.device)
        return self.word_embed(words_embedding),chars_embedding
    # sum of char embedding
    def words_embedding_v1(self,words):

            self.embed_model.get_model()
            # shape = [1,batch_size,padding_size*char_dim]
            words_embedding = []
            #shape = [1,batch_size,padding_size,char_dim]
            chars_embedding = []
            average_word_embeddings = []
            empty_embedding = [0.0]*self.char_dim
            for word in words:
                if word in self.embed_wdict.keys():
                   word_embedding = self.embed_wdict[word]
                else:
                   word_embedding = self.embed_model.word_embedding(word)
                   self.embed_wdict[word] = word_embedding
                words_embedding.append(word_embedding)

            for index,word_embed in enumerate(words_embedding):
                if len(word_embed)> self.padding_size:
                    chars_embedding.append(copy.deepcopy(word_embed[:self.padding_size]))
                else:
                    char_embedding = copy.deepcopy(word_embed)
                    gap_size = self.padding_size - len(word_embed)
                    for _ in range(gap_size): char_embedding.append(empty_embedding)
                    chars_embedding.append(char_embedding)
                np_word_embed = np.array(word_embed)
                # print(np_word_embed)
                np_word_embed = np.sum(np_word_embed,axis=0)
                average_word_embeddings.append(np_word_embed.tolist())
            words_embedding = torch.tensor(average_word_embeddings).to(self.device)
            chars_embedding = torch.tensor(chars_embedding).to(self.device)
            return self.word_embed(words_embedding),chars_embedding
class charcnn(nn.Module):
    def __init__(self, char_dim, kernel_widths =(2, 3), kernel_nums = (2, 2)):
        '''
        This module will take a list of "n-grams" = "filter widths"
        and create sum of N filters that will be concatenated
        '''
        super(charcnn, self).__init__()
        self.num_filters = sum(kernel_nums[index] * kernel_width for index,kernel_width in enumerate (kernel_widths))
        self.kernels = nn.ModuleList()
        # 添加卷积网络到ModuleList
        for index, kernel_width in enumerate(kernel_widths):
            self.kernels.append(nn.Conv2d(1, kernel_nums[index] * kernel_width, (kernel_width, char_dim), stride=(1, 1)))
    def forward(self,x):
        '''
        	input: [batch_size x,temporal x,max_char_length x,char dim]
        	output: [batch_size x,temporal x,num_filter]
        '''
        conv_results = [torch.sigmoid(kernel(x)).squeeze(-1) for kernel in self.kernels]
        x = [F.max_pool1d(conv_result, conv_result.size()[-1]) for conv_result in conv_results]
        x = torch.cat(x, dim=1)
        return x

class charembed:
    def __init__(self,sg,window,embed_dim,save_dir = '', iter = 100):
        self.sg = sg
        self.window = window
        self.embed_dim = embed_dim
        self.iter = iter
        self.model = None
        self.model_name = 'char2vec.model'
        self.save_dir = save_dir
        self.train_again = False
        self.model_path = path.join(save_dir,self.model_name)
        if not path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    def gen_model(self,words):
        '''
        :param words:
        :return:
        '''
        is_exist = path.exists(self.model_path)
        if is_exist and not self.train_again:
            self.model = word2vec.Word2Vec.load(self.model_path)
        else:
            self.model = word2vec.Word2Vec(sentences = words,sg= self.sg,iter=self.iter,window=self.window,size=self.embed_dim)
            self.model.save(self.model_path)
    def get_model(self):
        is_exist = path.exists(self.model_path)
        assert is_exist,'不能找到预训练模型，需要训练并保存模型！'
        self.model = word2vec.Word2Vec.load(self.model_path)

    def word_embedding(self, word):
        # a = self.model.wv
        return [self.model.wv[char].tolist() for char in word if char in self.model.wv]
        # return [self.model.wv.get_vector(char).tolist() for char in word if char in self.model.wv.vocab]


    def char_embedding(self, char):
        return self.model.wv.get_vector(char)

from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')

class VAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dim:int,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim =input_dim

        self.encoder = self.build_encoder(in_dim= input_dim,out_dim= hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        self.decoder = self.build_decoder(in_dim= hidden_dim,out_dim= latent_dim )

        self.final_layer = nn.Linear(self.latent_dim,self.input_dim)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def build_encoder(self,in_dim:int,out_dim:int):
        # Build Encoder
        encoder = nn.Sequential(
            nn.Linear(in_dim,out_dim),
            nn.LeakyReLU()
        )
        return encoder

    def build_decoder(self,in_dim,out_dim):
        #Build Decoder
        decoder = nn.Sequential(
            nn.Linear(in_dim,out_dim),
            nn.LeakyReLU()
        )
        return decoder

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)


        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu



    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        # [self.decode(z), input, mu, log_var]
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = kwargs['kld_weight']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        # loss = recons_loss + kld_weight * kld_loss
        loss = (1-kld_weight)*recons_loss + kld_weight * kld_loss
        # return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
        return loss,recons_loss
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

if __name__ == '__main__':
    dir = '../data/embed_parse/'
    print(path.exists(dir))
    sentences = word2vec.Text8Corpus('../data/embed_parse/chars.txt')

    embed = charembed(sg=0,window=3,embed_dim=10, save_dir=dir,iter= 100)
    embed.train_again = True
    embed.gen_model(sentences)
