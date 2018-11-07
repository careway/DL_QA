##stuff to download to get it to work
#!pip install torchtext spacy
#!python -m spacy download en
#!python -m spacy download fr
#wget https://s3.amazonaws.com/opennmt-models/iwslt.pt


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
devices =[0]

#parameters
justEvaluate = False
loadPreTrain = False
trainItNb = 10
BATCH_SIZE = 12000
validFreq = 5
previousEpochNb = 0
modelSavePath = ".Model/modelIWSLT.nn"

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

class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize

def training(model, optimizer, trainItNb, train_iter, valid_iter, validFreq, criterion, pad_idx, devices) :
    for epoch in range(trainItNb):
        model.train()
        #run_epoch computes the loss given the input optimizer function, which is here MultiGPULossCompute
        #if the argument optimizer of MultiGPULossCompute isn't none then it will also perform the backprop
        transformer.run_epoch((rebatch(pad_idx, b) for b in train_iter), 
                  model, 
                  MultiGPULossCompute(model.generator, criterion, 
                                      devices=devices, opt=optimizer))
        #validation
        if (epoch % validFreq == 0):
            evaluate(model, valid_iter, criterion, devices, optimizer, pad_idx)

def evaluate(model, valid_iter, criterion, devices, optimizer, pad_idx) :
    model.eval()
    loss = transformer.run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                     model, 
                     MultiGPULossCompute(model.generator, criterion, 
                     devices=devices, opt=None))
    print("loss : ", loss)

def saveModel(model, epoch, optimizer, batchSize, PATH):
    state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'batchSize' : batchSize
    }
    torch.save(state, PATH)

def loadModel(PATH):
    state = torch.load(PATH)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    batchSize.load_state_dict(state['batchSize'])
    epoch.load_state_dict(state['epoch'])

    return model,optimizer,batchSize,epoch

########################################################
#run code starts here
########################################################
    
#Load Data
print("Loading Data")
SRC,TGT,train,val,test, pad_idx = dataLoaderIWLST.loadDataIWLST()
print("Data Loaded")

if (loadPreTrain or justEvaluate) :
    print("Loading pre-trained network")
    model, model_opt, BATCH_SIZE, previousEpochNb = loadModel(modelSavePath)
else :
    print("initializing network")
    model = transformer.make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model_opt = transformer.NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

model.cuda()
criterion = transformer.LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.cuda()

print("Initializing iterators")
train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=transformer.batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=transformer.batch_size_fn, train=False)

#if more than one GPU, go parallel
model_par = nn.DataParallel(model, device_ids=devices)

if not (justEvaluate) :
    print("Starting training")
    training(model_par, model_opt, trainItNb, train_iter, valid_iter, validFreq, criterion, pad_idx)
    evaluate(model_par, valid_iter, criterion, devices, model_opt, pad_idx)

    print("Saving network")
    saveModel(model_par, previousEpochNb + trainItNb, model_opt, BATCH_SIZE, modelSavePath)
else :
    evaluate(model_par, valid_iter, criterion, devices, model_opt, pad_idx)