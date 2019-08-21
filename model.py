import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.autograd import Variable

class textRNN(nn.Module):
    def __init__(self, param):
        super(textRNN, self).__init__()
        self.param = param
        self.hidden_size=param['hidden_size']
        self.layers_num=param['layers_num']
        self.directions_num=param['directions_num']
         # self.batch_size=param['batch_size']
        self.vocab_size = param['vocab_size']
        self.embed_dim = param['embed_dim']
        self.class_num = param['class_num']

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim,padding_idx=1)
        self.lstm=nn.LSTM(self.embed_dim,self.hidden_size,batch_first=True)
        self.fc1=nn.Linear(self.hidden_size,128)
        self.fc2= nn.Linear(128, self.class_num)


    def forward(self, x):
        # x: (batch, sentence_length)
        x = self.embed(x)
        #X的维度应该是(len(sent),embed_dim)
        x = x.view(-1,20,self.embed_dim)
        out,_=self.lstm(x,None)
        #这里的out应该是hidden_size维度的
        out=F.log_softmax(self.fc1(out[:,-1,:]))
        out=F.log_softmax(self.fc2(out))

        return out


