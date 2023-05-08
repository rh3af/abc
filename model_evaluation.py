#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy
import numpy as np
import pandas as pd
import json

from src.model import build_model
from src.utils import convert_tensor, evaluate_model


# In[2]:


import numpy
tt = numpy.zeros((94129, 3194))
tt = tt.astype(np.float32)
tt = torch.from_numpy(tt)


# In[3]:


with open('./data/hyperparameter.json') as fp:
    hparam = json.load(fp)


# In[4]:


test_data = pd.read_pickle('./savepoints/0/test_data.pkl')
x = test_data[0]
y = test_data[1]


# In[5]:


SS_mat = pd.read_pickle('./data/structural_similarity_matrix.pkl')
TS_mat = pd.read_pickle('./data/target_similarity_matrix.pkl')
GS_mat = pd.read_pickle('./data/GO_similarity_matrix.pkl')


# In[6]:


mlb = pd.read_pickle('./data/mlb.pkl')
idx2label = pd.read_pickle('./data/idx2label.pkl')


# In[7]:


SS, TS, GS, y = convert_tensor(x, y, SS_mat, TS_mat, GS_mat, mlb, idx2label)
y = y.int().numpy()


# In[9]:


model = build_model(hparam)
model.load_model('./savepoints/0/model_checkpoint')
model.eval()


# In[10]:


with torch.no_grad():
    _, _, _, pred = model(SS, TS, GS)
    pred = torch.sigmoid(pred) > 0.5    
    pred = pred.int().numpy()


# In[11]:


acc, ma_rc, ma_pc, mi_rc, mi_pc = evaluate_model(y, pred)


# In[12]:


print(f'Accuracy: {acc}\nMacro recall: {ma_rc}\nMacro_precision: {ma_pc}\nMicro_recall: {mi_rc}\nMicro_precision: {mi_pc}')


# In[ ]:




