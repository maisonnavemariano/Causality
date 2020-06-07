#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from datetime import datetime
#BERT
import torch
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import  BertModel,BertForMaskedLM
import pickle
import numpy as np
import re

import threading

print(f'{datetime.now()} [ INFO  ] Finished imports, starting program.')


# In[2]:


# Data location
data_path = '/mnt/work/maiso/datasets/NYT'
print(f'{datetime.now()} [ INFO  ] Using {data_path} as the root data path.')


# In[3]:


print(f'{datetime.now()} [ INFO  ] Reading *.metadata filenames.')

## 1. Built the list with all the filenames:
files = []
for dirpath, _, filenames in os.walk(data_path):
    filenames = [filename for filename in filenames if filename.endswith('metadata')]
    for filename in filenames:
        file = os.path.join(dirpath, filename)
        files.append(file)

files.sort()
print(f'{datetime.now()} [ INFO  ] Finished reading *.metadata filenames from directory. Working with {len(files)} files.')


# In[4]:


xml_count = 0
metadata_count = 0
text_count = 0
lemmas_count = 0
events_count = 0
for dirpath, _, filenames in os.walk(data_path):
    xml_count += len([filename for filename in filenames if filename.endswith('.xml')])
    metadata_count += len([filename for filename in filenames if filename.endswith('metadata')])
    text_count += len([filename for filename in filenames if filename.endswith('.text')])
    lemmas_count += len([filename for filename in filenames if filename.endswith('.lemmas.p')])
    events_count += len([filename for filename in filenames if filename.endswith('.events.p')])
print(f'{datetime.now()} [ INFO  ] Curently are {xml_count} XMLs, {metadata_count} metadatas,{text_count} texts, {lemmas_count} lemmas and {events_count} event files.')

if events_count>0:
    print(f'{datetime.now()} [WARNING] Some files are goint to be overwritten (*.events.p)')


# In[5]:


print(f'{datetime.now()} [ INFO  ] Loading BERT class')
class BERT_Embeddings(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained('bert-base-uncased')

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()


    def get_BERT_Embeddings(self,text, tokens):
        matrix = np.zeros(shape=(1,len(tokens),768)) # 768 last 4 layer added.

        tokenized_text = ['[CLS]']
        
        for ini,fin in tokens:
            word_tokens = self.tokenizer.tokenize(text[ini:fin])
            tokenized_text +=word_tokens
        tokenized_text += ['[SEP]']
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
            
        token_embeddings = [] 
        layer_i = 0
        batch_i = 0
        token_i = 0

        # For each token in the sentence...
        for token_i in range(len(tokenized_text)):
            hidden_layers = [] 

            # For each of the 12 layers...
            for layer_i in range(len(encoded_layers)):

                # Lookup the vector for `token_i` in `layer_i`
                vec = encoded_layers[layer_i][batch_i][token_i]

                hidden_layers.append(vec)

            token_embeddings.append(hidden_layers)

        summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] # [number_of_tokens, 768]

        inicio=1
        for idx,token in enumerate(tokens):
                ini,fin = token
                individual_tokenized = self.tokenizer.tokenize(text[ini:fin])
                cantidad_embeddings = len(individual_tokenized)
                group_tokenized = tokenized_text[inicio:inicio+cantidad_embeddings]
                                
                assert all([t1==t2 for t1,t2 in zip(individual_tokenized,group_tokenized)]),  '{} {} {}'.format(
                                 text[ini:fin],
                                 individual_tokenized,
                                 group_tokenized
                             )

                if cantidad_embeddings>0:
                    for embedding in summed_last_4_layers[inicio:inicio+cantidad_embeddings]:
                        matrix[0,idx,:]+=embedding.numpy()
                    matrix[0,idx,:] = matrix[0,idx,:]/cantidad_embeddings
                    inicio+=cantidad_embeddings

        return matrix
print(f'{datetime.now()} [ INFO  ] BERT class loaded.')


# **We need the entity vocab for the next step. We first load model and vocab**

# In[ ]:


from modules.ed import get_data, load_model_from_disk, Generator, train_val_split, train_model, get_model, sensitivity, specificity, f1_score


# In[7]:


from modules.ed import get_data, load_model_from_disk, Generator, train_val_split, train_model, get_model, sensitivity, specificity, f1_score

data_path='/mnt/work/maiso/python3.workspace/Causality/data/ed/'
model_path = '/mnt/work/maiso/python3.workspace/Causality/models/best_model_15.h5'

print(f'{datetime.now()} [ INFO  ] Loading data from files (testing_sents.p and training_sents.p) on folder: {data_path}')
      
    
x, y, ent2index = get_data(data_path)
train_data, val_data = train_val_split((x,y))

print(f'{datetime.now()} [ INFO  ] Loading model from: {model_path}')
# model = load_model_from_disk(model_path)
model = get_model(len(ent2index), hidden_units_count=15)

model.compile(loss='binary_crossentropy', 
          optimizer='adam', 
          metrics=['accuracy',sensitivity,specificity, f1_score])

model.load_weights(model_path)

model._make_predict_function() # para read-only y evitar problemas de sincronización con multithreading

print(f'{datetime.now()} [ INFO  ] evaluating model on validation_data')
x_val, y_val = val_data
rta = model.evaluate(Generator(list(zip(x_val, y_val))), verbose=0)

# for metric, value in zip(['loss']+[m.name for m in model.metrics], rta):
#     print(f'{datetime.now()} [ INFO  ] {metric:12}: {value}')
for metric,value in zip(model.metrics_names, rta):
    print('{}: {:5.4f}'.format(metric, value))


# In[21]:


# Inputs: 
bert = BERT_Embeddings()
# bert_lock = threading.Lock()

    
# model_lock = threading.Lock()
tuplas_re = re.compile('\(([0-9]*?), ([0-9]*?)\)')
entities_re = re.compile("'([^']*)'")

print(f'{datetime.now()} [ INFO  ] Loading BERT model.')
print(f'{datetime.now()} [ INFO  ] BERT model loaded.')

def process_file(file):



    # read file (metadata + text)
    text = open(file[:-9]+'.text', 'r' ).read()
    lemmas = pickle.load(open(file[:-9]+'.lemmas.p', 'rb' ))

    metadata = open(file,'r').read().splitlines()
    id_ = metadata[0][4:] #len('ID: ') == 4
    date = metadata[1][6:] # len('DATE: ') == 6

    # process file
    tokens = tuplas_re.findall(metadata[3])
    sentences = tuplas_re.findall(metadata[2])

    tokens = [(int(ini),int(fin)) for ini, fin in tokens]
    sentences = [(int(ini),int(fin)) for ini, fin in sentences]

    entities = entities_re.findall(metadata[4])

    idx = 0
    old_idx = 0
    sent_idx=0
    for ini,fin in sentences:
        while idx<len(tokens) and tokens[idx][0]<fin:
            idx+=1
        tokens_in_sentence = tokens[old_idx:idx]
        lemmas_in_sentence = lemmas[old_idx:idx]
        entities_in_sentence = entities[old_idx:idx]
        old_idx = idx

#         with bert_lock:
        xbert = bert.get_BERT_Embeddings(text, tokens_in_sentence )

        xsentbert = np.average(xbert, axis=1) *np.ones(shape=(1,len(xbert[0,:,0]),768))
        xents = np.zeros(shape=(1,len(tokens_in_sentence)))


        for idx,entity in enumerate(entities_in_sentence):
            if entity in ent2index:
                xents[0,idx] = ent2index[entity]
            else:
                xents[0,idx] = ent2index['[PAD]']

        x = [xbert, xsentbert, xents]
#         with model_lock:
        y_pred = model.predict(x)
        events = []

        for j in range(len(tokens_in_sentence)):
            pred = y_pred[0,j,0]
            if pred>0.5:
                bert_vec = xbert[0,j,:]
                token = tokens_in_sentence[j]


                event = {
                    'file_id':id_,
                    'trigger': text[token[0]:token[1]],
                    'idx': sent_idx,
                    'bert': bert_vec,
                    'conf': pred,
                    'lemma': lemmas_in_sentence[j],
                    'date': date
                }
                events.append(event)
        sent_idx+=1

        pickle.dump(events ,open(file[:-9]+'.events.p', 'wb' ))
    return file



print(f'{datetime.now()} [ INFO  ] process_file defined. Uses: metadata, lemmas and text for building the event.p files.')


# In[ ]:


import concurrent.futures

max_workers=16 
chunksize = int(len(files)/(max_workers))

print(f'{datetime.now()} [ INFO  ] Starting the concurrent processing of all the files ({len(files)}) using max_workers={max_workers} and chunksize={chunksize}.')
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(executor.map(process_file, files, chunksize=chunksize))
    
print(f'{datetime.now()} [  OK   ] Finished!')
