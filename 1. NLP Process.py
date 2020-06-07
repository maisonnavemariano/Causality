#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import re
import spacy
from multiprocessing import Pool
from bs4 import BeautifulSoup
import pickle
from datetime import datetime
print(f'{datetime.now()} [ INFO  ] Finished imports, starting program.')

# Data location
data_path = '/mnt/work/maiso/datasets/NYT'
print(f'{datetime.now()} [ INFO  ] Using {data_path} as the root data path.')
# Nro of Process defined to be equal to nro of cores.
nro_of_process = 16

print(f'{datetime.now()} [ INFO  ] Running with {nro_of_process} no. of process.')


print(f'{datetime.now()} [ INFO  ] The following program create the files: *.metadata, *.text and *.lemmas.p')

xml_count = 0
metadata_count = 0
text_count = 0
lemmas_count = 0
for dirpath, _, filenames in os.walk(data_path):
    xml_count += len([filename for filename in filenames if filename.endswith('.xml')])
    metadata_count += len([filename for filename in filenames if filename.endswith('metadata')])
    text_count += len([filename for filename in filenames if filename.endswith('.text')])
    lemmas_count += len([filename for filename in filenames if filename.endswith('.lemmas.p')])
print(f'{datetime.now()} [ INFO  ] Curently are {xml_count} XMLs, {metadata_count} metadatas,{text_count} texts and {lemmas_count} lemmas')

if text_count>0 or lemmas_count>0 or metadata_count>0:
    print(f'{datetime.now()} [WARNING] Some files are goint to be overwritten')


# In[4]:


## 1. Built the list with all the filenames:
print(f'{datetime.now()} [INFO] Reading .xml filenames.')
files = []
for dirpath, _, filenames in os.walk(data_path):
    filenames = [filename for filename in filenames if filename.endswith('xml')]
    for filename in filenames:
        file = os.path.join(dirpath, filename)
        files.append(file)

files.sort()
print(f'{datetime.now()} [INFO] Finished reading .xml filenames from directory. Working with {len(files)} files.')


# In[7]:


print(f'{datetime.now()} [INFO] Compiling regex patterns and defining generator of articles')
article_text_ =  re.compile('<block class="full_text".*?>(.*?)</block>',re.DOTALL)
article_date_ = re.compile('<pubdata date.publication="([0-9]{8})T[0-9]{6}".*?/>', re.DOTALL)
article_id_ = re.compile('<doc-id id-string="([0-9]{1,7})"/>')
nlp = spacy.load('en_core_web_sm', disable=['textcat'])


articles = (open(file,'r').read() for file in files)



def NLPProcess(article):
    text = article_text_.findall(article)
    text = text[0] if len(text)==1 else ''
    text = BeautifulSoup(text, 'html.parser').get_text()
    spacy_doc = nlp(text)
    
    date = article_date_.findall(article)[0]
    id_ = article_id_.findall(article)

    assert len(id_) == 1, 'The article has more than one identifier'
    id_ = f'{int(id_[0]):07}'

    writer = open(f'{data_path}/{date[:4]}/{date[4:6]}/{date[6:8]}/{id_}.metadata', 'w')
    
    # tokens
    tokens = []
    if len(spacy_doc)!= 0:
        tokens = [(token.idx, token.idx+len(token.text)) for token in spacy_doc]
    
    # sents
    sents = []
    if len(spacy_doc)!= 0:
        inis  = [spacy_doc[sent.start].idx for sent in spacy_doc.sents]
        fins  = inis[1:]+[spacy_doc[-1].idx+len(spacy_doc[-1].text)]
        sents = list(zip(inis,fins))
    
    # entities
    entities = []
    if len(spacy_doc)!= 0:
        #entities = [token for token in spacy_doc if token.ent_iob_!='O']
        entities = [token for token in spacy_doc]
        entities = [ f'{token.ent_iob_}-{token.ent_type_}' for token in entities]
    
    
    
    if len(spacy_doc)!= 0:
        lemmas = [token.lemma_ for token in spacy_doc]
    else:
        lemmas = []
        
    writer.write(f'ID: {id_}\n')
    writer.write(f'DATE: {date}\n')
    writer.write(f'SENTS: {sents}\n')
    writer.write(f'TOKEN: {tokens}\n')
    writer.write(f'ENTITIES: {entities}\n')
    
    writer.close()
    
    
    writer = open(f'{data_path}/{date[:4]}/{date[4:6]}/{date[6:8]}/{id_}.text', 'w')
    writer.write(text)
    writer.close()
    
    pickle.dump(lemmas, open(f'{data_path}/{date[:4]}/{date[4:6]}/{date[6:8]}/{id_}.lemmas.p', 'wb'))

    

chunksize = int(len(files)/(3*nro_of_process)) # ~3 chunks per process.
print(f'{datetime.now()} [INFO] Starting multiprocessing with chunksize={chunksize}')
with Pool(processes=nro_of_process) as pool:
    pool.map(NLPProcess, articles,chunksize=chunksize)
    
print(f'{datetime.now()} [  OK   ] Finished!')


# In[ ]:




