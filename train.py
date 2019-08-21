from torch.autograd import Variable
import torch
import os
import torch.nn as nn
import numpy as np
import time
from model import textRNN
import sen2inds
from textRNN_data import textRNN_dataLoader,get_valdata

word2ind, ind2word = sen2inds.get_worddict('wordLabel.txt')
label_w2n, label_n2w = sen2inds.read_labelFile('label.txt')

textRNN_param = {
    'vocab_size': len(word2ind),
    'embed_dim': 60,
    'class_num': len(label_w2n),
    "kernel_num": 16,
    "hidden_size":256,
    "layers_num":1,
    "directions_num":1,
}
dataLoader_param = {
    'batch_size': 64,
    'shuffle': True,
}


def main():
    #init net
    print('init net...')
    net = textRNN(textRNN_param)
    weightFile = 'weight.pkl'
    print('init dataset...')
    dataLoader =textRNN_dataLoader(dataLoader_param)
    valdata =get_valdata()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    log = open('log_{}.txt'.format(time.strftime('%y%m%d%H')), 'w')
    log.write('epoch step loss\n')
    log_test = open('log_test_{}.txt'.format(time.strftime('%y%m%d%H')), 'w')
    log_test.write('epoch step test_acc\n')
    print("training...")
    for epoch in range(15):
        for i, (clas, sentences) in enumerate(dataLoader):
            optimizer.zero_grad()
            sentences = sentences.type(torch.LongTensor)
            clas = clas.type(torch.LongTensor)
            out = net(sentences)
            loss = criterion(out, clas)
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
                data = str(epoch + 1) + ' ' + str(i + 1) + ' ' + str(loss.item()) + '\n'
                log.write(data)
        print("save model...")
        torch.save(net.state_dict(), weightFile)
        print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())


if __name__ == "__main__":
    main()
