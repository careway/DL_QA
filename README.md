This our implementation of a transformer based text translator, as described in Attention is all you need by Vaswani et al.

transformer.py : contains everything to create a transformer network
training.py : contains the methods for training the network and testing a pre-trained model
dataLoaderIWLST.py : contains the method for loading the IWLST database
python_job.sh : run file

To get started open the transformer.py file in your IDE, and tweak the parameters at the top of the file to your liking :
justEvaluate : if true, this will load the pre trained model at "modelSavePath" and eavluate it only
loadPreTrain : if true, this will load the pre trained model at "modelSavePath" and train it some more
trainItN : max number of training iterations (if greater than the total number of sample, training will stop when the last sample is reached)
validItNb : number of validation/evaluation iterations 
BATCH_SIZE : batch size
validFreq : unused
previousEpochNb : unused
modelSavePath : path to where to load and save the model at
testSentence : custom test sentence for evaluation.
testTranslation : target translation for the above mentionned custom translation

Once this is done, simply run python_job.sh in your linux terminal


Alternatively, if you have enough memory, you can run everything from the notebook TranslationFREN.ipynb
