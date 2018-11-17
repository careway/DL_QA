from torchtext import data, datasets
import spacy

def loadDataIWLST():
    print("Doing spacy load")
    spacy_fr = spacy.load('fr')
    spacy_en = spacy.load('en')
    print("Spacy load completed")

    def tokenize_fr(text):
        return [tok.text for tok in spacy_fr.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    print("Creating data Fields")
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_fr, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)
    
    MAX_LEN = 100

    print("Splitting")
    train, val, test = datasets.IWSLT.splits(
        exts=('.fr', '.en'), fields=(SRC, TGT), 
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
            len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2

    print("Building vocabulary")
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    pad_idx = TGT.vocab.stoi[BLANK_WORD]

    return SRC,TGT,train,val,test, pad_idx
