# -*- coding: utf-8 -*-
"""training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QH-lSc-hIJCb7qZnURuphOB0Mty88JaZ
"""

##stuff to download to get it to work
#!pip install torchtext spacy
#!python -m spacy download en
#!python -m spacy download fr
#!wget https://s3.amazonaws.com/opennmt-models/iwslt.pt


#import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import math, copy, time
from torch.autograd import Variable
from torchtext import data#, datasets

import transformer
import dataLoaderIWLST 

##visualization thingy
#import matplotlib.pyplot as plt
#import seaborn
#seaborn.set_context(context="talk")
#%matplotlib inline

#create table of different GPUSs
#print("current device index", torch.cuda.current_device())
#current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print("number of available cuda devices : ", torch.cuda.device_count())
#devices = range(torch.cuda.device_count())
print("WARNING : As of now MultiGpu is not supported so all options using MultiGPU will be downgraded to single GPU")
#device = torch.cuda.current_device() 
device = torch.device(torch.cuda.current_device())
print(device)
#parameters
justEvaluate = False
loadPreTrain = False
trainItNb = 2000
validItNb = 200
BATCH_SIZE = 100
validFreq = 5
previousEpochNb = 0
modelSavePath = "Model/modelIWSLT.nn"

"""**Load Data **"""

#Load Data
print("Loading Data")
SRC,TGT,train,val,test, pad_idx = dataLoaderIWLST.loadDataIWLST()
print("Data Loaded")

"""**Define Iterator Methods**"""

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return transformer.Batch(src, trg, pad_idx)


"""Visualisation """
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

def draw(data, x, y, ax):
    seaborn.heatmap(data.cpu(), 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax)
    #plt.imshow(data, cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0,)

"""load/save"""

def saveModel(model,model_opt ,epoch, batchSize, PATH):
    state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'batchSize' : batchSize
    }
    torch.save(state, PATH)

def loadModel(PATH, SRC, TGT):
    state = torch.load(PATH)
    
    model = transformer.make_model(len(SRC.vocab), len(TGT.vocab))
    model.load_state_dict(state['state_dict'])
    
    batchSize = state['batchSize']
    
    epoch = state['epoch']

    return model,batchSize,epoch

"""**Initialize model, optimizer, criterion and iterators**"""

if (loadPreTrain or justEvaluate) :
    print("Loading pre-trained network")
    model, BATCH_SIZE, previousEpochNb = loadModel(modelSavePath, SRC, TGT)
else :
    print("initializing network")
    model = transformer.make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    
model_opt = transformer.NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=0.001, betas=(0.9, 0.98), eps=1e-8))    
model.cuda()
#criterion = transformer.LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
criterion = nn.CrossEntropyLoss()
#criterion.cuda()



print("Initializing iterators")
#train_iter = MyIterator(train, batch_size=BATCH_SIZE, device = device,
#                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
#                        batch_size_fn=transformer.batch_size_fn, train=True)
#valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device = device,
#                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
#                        batch_size_fn=transformer.batch_size_fn, train=False)

train_iter = data.BucketIterator(train, batch_size=BATCH_SIZE, device=device, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), train=True)
valid_iter = data.BucketIterator(val, batch_size=BATCH_SIZE, device=device, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), train=False)


#if more than one GPU, go parallel
#model_par = nn.DataParallel(model, device_ids = devices)

"""**Define Loss**"""
class SingleGPULossCompute:
    "A single-gpu loss compute and train function."
    def __init__(self, generator, criterion, device, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        #self.device = device
        #self.chunk_size = chunk_size
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        norm = norm.float()
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm.item()
        
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(transformer.subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def training(model, optimizer, trainItNb, train_iter, valid_iter, validFreq, criterion, pad_idx, device) :
    model.train()
    transformer.run_epoch((rebatch(pad_idx, b) for b in train_iter), 
        model, 
        SingleGPULossCompute(model.generator, criterion, 
                             device=device, opt=optimizer), trainItNb)    

def evaluate(model, valid_iter, criterion, device, optimizer, pad_idx, validItNb) :
    model.eval()
    print(pad_idx)
    loss = transformer.run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                     model, 
                     SingleGPULossCompute(model.generator, criterion, 
                     device=device, opt=None), validItNb)
    j = 0       
    for i, batch in enumerate(valid_iter):
        j += 1
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask, 
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        
        print("Source:")    
        src_txt = "<s> "
        for i in range(1, batch.src.size(0)):
            sym = SRC.vocab.itos[batch.src.data[i, 0]]
            if sym == "</s>": break
            src_txt += sym + " "
        print(src_txt.encode("utf-8"))
        
        print("Translation:")
        trans = "<s> "
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>": break
            trans += sym + " "
        print(trans.encode("utf-8"))
        print("Target:")
        tgt_print = ""
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            tgt_print += sym + " "
        print(tgt_print.encode("utf-8"))
    #loss = transformer.run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
    #                 model, 
    #                 MultiGPULossCompute(model.generator, criterion, 
    #                 devices=devices, opt=None))
    print("loss : ", loss)
    
    """Translate default sentence"""
    print("Source:")
    print("Le cheval est grand .")
    sent = """Le cheval est grand .""".split()  
    src = torch.LongTensor([[SRC.vocab.stoi[w] for w in sent]])
    src = Variable(src).cuda()
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask,max_len=60, 
                        start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:")
    trans = "<s> "
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        trans += sym + " "
    print(trans.encode("utf-8"))                   
    print("Target :")
    print("The horse is big .")
    
    print("Plotting and exporting graphs")
    
    tgt_sent = trans.split()   
    print(len(tgt_sent))
    print(len(sent))
    
    for layer in range(1, 6, 2):
        num = str(layer+1)
        fig, axs = plt.subplots(1,4, figsize=(20, 10))
        print("Encoder Layer", layer+1)
        for h in range(4):
            draw(model.encoder.layers[layer].self_attn.attn[0, h].data[:len(sent), :len(sent)], 
                sent, sent if h ==0 else [], ax=axs[h])
        plt.show()
        plt.savefig('encoderlayer_' + num + ' .svg')

    for layer in range(1, 6, 2):
        num = str(layer+1)
        fig, axs = plt.subplots(1,4, figsize=(20, 10))
        print("Decoder Self Layer", layer+1)
        for h in range(4):
            draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)], 
                tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
        plt.show()
        plt.savefig('DecoderSelfLayer_' + num + ' .svg')
    print("Decoder Src Layer", layer+1)
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    for h in range(4):
        draw(model.decoder.layers[layer].src_attn.attn[0, h].data[:len(tgt_sent), :len(sent)], 
            sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()
    plt.savefig('DecoderSrcLayer_' + num + ' .svg')

    

"""**Run training/Evaluate**""" 

#criterion = nn.CrossEntropyLoss()
#criterion.cuda()

if not (justEvaluate) :
    print("Starting training")
    #training(model_par, model_opt, trainItNb, train_iter, valid_iter, validFreq, criterion, pad_idx, devices)
    #evaluate(model_par, valid_iter, criterion, devices, model_opt, pad_idx)
   
    training(model, model_opt, trainItNb, train_iter, valid_iter, validFreq, criterion, pad_idx, device)
    evaluate(model, valid_iter, criterion, device, model_opt, pad_idx, validItNb)

    print("Saving network")
    #saveModel(model_par, previousEpochNb + trainItNb, model_opt, BATCH_SIZE, modelSavePath)
    saveModel(model,model_opt, previousEpochNb + trainItNb, BATCH_SIZE, modelSavePath)
else :
    evaluate(model, valid_iter, criterion, device, model_opt, pad_idx,validItNb)
    #evaluate(model_par, valid_iter, criterion, devices, model_opt, pad_idx)



