#!/usr/bin/env python
# coding: utf-8

# In[27]:


import sys
import sys, getopt
from datetime import datetime


argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv,"",["training=",
                                        "epochs=",
                                        "model_path=",
                                        "hidden_units_count="
                                       ])
    
    for opt, arg in opts:
        if opt == "--training":
            training = arg=="True"
        if opt == "--epochs":
            epochs = int(arg)
        if opt == "--model_path":
            model_path = arg
        if opt == "--hidden_units_count":
            hidden_units_count = int(arg)
            
            
except getopt.GetoptError:
    training = False
    epochs = 1
    model_path = '/mnt/work/maiso/python3.workspace/Causality/models/best_model.h5'
    hidden_units_count = 15
    print(f'{datetime.now()} [WARNING] Bad or missing parameters. Defined to default:')

print(f'{datetime.now()} [ INFO  ] Parameters:' )
print(f'\t\t training           = {training}')
print(f'\t\t epochs             = {epochs}')
print(f'\t\t model_path         = {model_path}')
print(f'\t\t hidden_units_count = {hidden_units_count}')


# In[22]:


from modules.ed import get_data, load_model_from_disk, Generator, train_val_split, train_model, get_model, sensitivity, specificity, f1_score


# In[23]:


data_path='/mnt/work/maiso/python3.workspace/Causality/data/ed/'


# In[24]:


print(f'{datetime.now()} [ INFO  ] Loading data from files (testing_sents.p and training_sents.p) on folder: {data_path}')
    
x, y, ent2index = get_data(data_path)
train_data, val_data = train_val_split((x,y))


# In[25]:


print(f'{datetime.now()} [ INFO  ] Generating model')
model = get_model(len(ent2index), hidden_units_count=hidden_units_count)


# In[ ]:


if not training:
    model.load_weights(model_path)

model.compile(loss='binary_crossentropy', 
          optimizer='adam', 
          metrics=['accuracy',sensitivity,specificity, f1_score])

if training:
    train_model(model, train_data, val_data, model_path, epochs=epochs)


# In[ ]:


print(f'{datetime.now()} [ INFO  ] evaluating model on validation_data')
x_val, y_val = val_data
rta = model.evaluate(Generator(list(zip(x_val, y_val))), verbose=0)

for metric,value in zip(model.metrics_names, rta):
    print('{}: {:5.4f}'.format(metric, value))

