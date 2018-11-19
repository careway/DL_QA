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
BATCH_SIZE = 10
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

def saveModel(model, epoch, batchSize, PATH):
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
    
    batchSize = 0
    batchSize.load_state_dict(state['batchSize'])
    
    epoch = 0
    epoch.load_state_dict(state['epoch'])

    return model,batchSize,epoch

"""**Initialize model, optimizer, criterion and iterators**"""

if (loadPreTrain or justEvaluate) :
    print("Loading pre-trained network")
    model, model_opt, BATCH_SIZE, previousEpochNb = loadModel(modelSavePath, SRC, TGT)
else :
    print("initializing network")
    model = transformer.make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    
model_opt = transformer.NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-8))    
model.cuda()
criterion = transformer.LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
#criterion = nn.CrossEntropyLoss()
#criterion.cuda()



print("Initializing iterators")
train_iter = MyIterator(train, batch_size=BATCH_SIZE, device = device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=transformer.batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device = device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=transformer.batch_size_fn, train=False)

#if more than one GPU, go parallel
#model_par = nn.DataParallel(model, device_ids = devices)

"""**Define Loss**"""

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
            normalize = normalize.float()
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

class SingleGPULossCompute:
    "A single-gpu loss compute and train function."
    def __init__(self, generator, criterion, device, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.device = device
        #self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
                
        out1 = out.to(device)
        targets = targets.to(device)
        
        #apply generator
        gen = self.generator.forward(out1).to(device)
                
        #compute loss by applying criterion
       
        loss = self.criterion(gen[:, :, :].contiguous().view(-1, gen.size(-1)), targets[:, :].contiguous().view(-1))


        normalize = normalize.float()
        l = loss.sum()[0] / normalize
        total = l.data[0]
        
        if self.opt is not None:
            #backprop to transformer output
            loss.backward()            

            o2 = out1.grad.clone()
            
            out1.backward(gradient=o2)
            
            #backprop through transformer
            #grad1 = gen.grad.data.clone()
            #gen.backward(gradient = grad1)
            #step optimizer and zero_grad()
            self.opt.step()
            self.opt.optimizer.zero_grad()
            out=out1
        
        return total*normalize      
    
def training(model, optimizer, trainItNb, train_iter, valid_iter, validFreq, criterion, pad_idx, device) :
    model.train()
    transformer.run_epoch((rebatch(pad_idx, b) for b in train_iter), 
        model, 
        SingleGPULossCompute(model.generator, criterion, 
                             device=device, opt=optimizer), trainItNb)    

def evaluate(model, valid_iter, criterion, device, optimizer, pad_idx, validItNb) :
    model.eval()
    loss = transformer.run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                     model, 
                     SingleGPULossCompute(model.generator, criterion, 
                     device=device, opt=None), validItNb)
    #loss = transformer.run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
    #                 model, 
    #                 MultiGPULossCompute(model.generator, criterion, 
    #                 devices=devices, opt=None))
    
    print("loss : ", loss)

"""**Run training/Evaluate**"""



criterion = nn.CrossEntropyLoss()
#criterion.cuda()

if not (justEvaluate) :
    print("Starting training")
    #training(model_par, model_opt, trainItNb, train_iter, valid_iter, validFreq, criterion, pad_idx, devices)
    #evaluate(model_par, valid_iter, criterion, devices, model_opt, pad_idx)
   
    training(model, model_opt, trainItNb, train_iter, valid_iter, validFreq, criterion, pad_idx, device)
    evaluate(model, valid_iter, criterion, device, model_opt, pad_idx, validItNb)

    print("Saving network")
    #saveModel(model_par, previousEpochNb + trainItNb, model_opt, BATCH_SIZE, modelSavePath)
    saveModel(model, previousEpochNb + trainItNb, BATCH_SIZE, modelSavePath)
else :
    evaluate(model, valid_iter, criterion, device, model_opt, pad_idx)
    #evaluate(model_par, valid_iter, criterion, devices, model_opt, pad_idx)



