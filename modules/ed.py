##
## In []
## 
import os
import pickle
import spacy
import numpy as np

def get_data(data_path):
    """
    Function that takes a path to the data location and reads 

    Parameters
    ----------
    data_path : str
        A string that indicates the location of the dataset in disk. The directory is expected to have two pickles to load: 
            1. training_sents.p
            2. testing_sents.p

    Returns
    -------
    tuple x, y, vocab
        x and y are lists to the inputs and label information 
    """
    training_sents_path = os.path.join(data_path,'training_sents.p') 
    testing_sents_path = os.path.join(data_path,'testing_sents.p')
    
    oraciones = pickle.load(open(training_sents_path,'rb'))
    oraciones += pickle.load(open(testing_sents_path,'rb'))
    
    # entities vocab
    nlp = spacy.load('en_core_web_sm', disable=['tagger','parser','textcat'])
    entity_vocab = set()
    for oracion in oraciones:
        oracion['spacy_doc'] = nlp(oracion['texto'])
        assert all([token.idx==i and (token.idx+len(token.text))==f for (i,f),token in zip(oracion['tokens'], oracion['spacy_doc'])]), '{}!={}'.format(token.i,i)
        e = [token.ent_iob_+'-'+token.ent_type_ for token in oracion['spacy_doc']]
        entity_vocab.update(set(e))


    entity_vocab.add('[PAD]')
    entity_vocab = list(entity_vocab)
    ent2index = dict([(ent,index) for index,ent in enumerate(entity_vocab)])
    
    xberts = [np.array(oracion['bert'])[np.newaxis,:,:] for oracion in oraciones]
    xsentberts = [np.average(xbert, axis=1) *np.ones(shape=(1,len(xbert[0,:,0]),768)) for xbert in xberts]
    xents = [np.zeros(shape=(1,len(oracion['spacy_doc']))) for oracion in oraciones]                   
     
    for idx,xent in enumerate(xents):
        for idx,token in enumerate(oraciones[idx]['spacy_doc']):
            xent[0,idx] = ent2index[token.ent_iob_+'-'+token.ent_type_]
       
    
    x = list(zip(xberts,xsentberts, xents)) 
    x = [list(elem) for elem in x]
    y = [np.array(oracion['etiquetas'])[np.newaxis,:,np.newaxis] for oracion in oraciones]
    return x,y, ent2index

##
## In []
## 
from keras.models import Sequential,Model
from keras.layers import Dense, Embedding,Input, Dropout, Bidirectional, LSTM# Create the model
import numpy as np
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Lambda
import tensorflow as tf
from keras.backend import expand_dims
from keras.layers.merge import concatenate
import tensorflow
from keras.models import model_from_json
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.callbacks import K




def get_model(entity_vocab_size, hidden_units_count=7):
    """
    Function that ...

    Parameters
    ----------
    entity_vocab_size : str
        A string that indicates ... 
    hidden_units_count : int
        A string that indicates ... 
       

    Returns
    -------
    tuple x, y, vocab
    """
    
    bert_input = Input((None,768))
    bert_sent_input = Input((None,768))

    input_entities = Input((None,))
    embedding_entities  = Embedding(entity_vocab_size, 10)(input_entities)

    
    merged = concatenate([bert_input, bert_sent_input, embedding_entities])
    dropout_1 = Dropout(0.1)(merged)
    lstm_1 = Bidirectional(LSTM(hidden_units_count,return_sequences=True,activity_regularizer=l1_l2(0.001,0.001)))(dropout_1)
    dense_out = Dense(1, activation='sigmoid')(lstm_1)

    
    inputs = [bert_input, bert_sent_input, input_entities]
    model = Model(inputs=inputs, outputs=[dense_out])
    
    return model

##
## In []
## 

from keras.utils import Sequence
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    sens =  true_positives / (possible_positives + K.epsilon())
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    spec = true_negatives / (possible_negatives + K.epsilon())
    return 2*((sens*spec)/(sens+spec+K.epsilon()))

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

class Generator(Sequence):
    
    def __init__(self, label_data):
        self.label_data = label_data        
    def __getitem__(self, idx):
        return self.label_data[idx] # tuple x,y
    def __len__(self):
        return len(self.label_data)
    def on_epoch_end(self):
        np.random.shuffle(self.label_data)
        
my_seed = 2
np.random.seed(my_seed)

from keras.models import load_model

def load_model_from_disk(model_path):
    """
    Function that ...

    Parameters
    ----------
    entity_vocab_size : str
        A string that indicates ... 
    hidden_units_count : int
        A string that indicates ... 
       

    Returns
    -------
    tuple x, y, vocab
    """
    model = load_model(model_path, custom_objects={'sensitivity': sensitivity, 'specificity':specificity, 'f1_score':f1_score})
    return model

def train_model(model, train_data, val_data, best_model_path, epochs=2000 ):
    """
    Function that ...

    Parameters
    ----------
    entity_vocab_size : str
        A string that indicates ... 
    hidden_units_count : int
        A string that indicates ... 
       

    Returns
    -------
    tuple x, y, vocab
    """
    x_train, y_train = train_data
    x_val, y_val = val_data
    mc = ModelCheckpoint(best_model_path, monitor='val_f1_score', save_best_only=True,mode='max')
    
    #Restore best model Lookup
    es = EarlyStopping(monitor='val_f1_score', mode='max', verbose=1, patience=400, restore_best_weights=True)
    
    history = model.fit_generator(Generator([(x,y) for x,y in zip(x_train,y_train)]), 
                               epochs=epochs, 
                                  validation_data = Generator([(x,y) for x,y in zip(x_val,y_val)]),
                               callbacks=[mc, es]
                                 )
    return history
    
def train_val_split(data):
    """
    Function that ...

    Parameters
    ----------
    entity_vocab_size : str
        A string that indicates ... 
    hidden_units_count : int
        A string that indicates ... 
       

    Returns
    -------
    tuple x, y, vocab
    """
    train_idx_path = '/mnt/work/maiso/python3.workspace/Causality/data/cache/train_indexes.p'
    x, y = data
    
    if os.path.exists(train_idx_path):
        print('[  OK   ] Loading training indexes.')
        train_idx = pickle.load(open(train_idx_path,'rb'))
    else:
        print('[WARNING] Computing training indexes.')
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        train_idx = set(indices[:int(len(indices)*.80)])
        pickle.dump(train_idx, open(train_idx_path, 'wb'))

    x_train, y_train = [e for idx,e in enumerate(x) if idx in train_idx],\
                    [e for idx,e in enumerate(y) if idx in train_idx]
    x_val, y_val = [e for idx,e in enumerate(x) if not idx in train_idx],\
                            [e for idx,e in enumerate(y) if not idx in train_idx]
    train_data, val_data = (x_train, y_train),(x_val, y_val)
    return train_data, val_data
