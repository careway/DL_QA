#!/bin/sh



module load cuda/9.2
module load python3/3.6.2



#python3 -m venv train_model

#source train_model/bin/activate

pip3 install --user os
pip3 install --user matplotlib
pip3 install --user torch torchvision
pip3 install --user torchtext spacy
pip3 install --user --upgrade pip
pip3 install --user seaborn
python3 -m spacy download en --user
python3 -m spacy download fr --user
#wget https://s3.amazonaws.com/opennmt-models/iwslt.pt 

CUDA_VISIBLE_DEVICES=0 python3 training.py
