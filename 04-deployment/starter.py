#!/usr/bin/env python
# coding: utf-8

# In[18]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[19]:


get_ipython().system('python -V')


# In[20]:


import pickle
import pandas as pd


# In[21]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[22]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[23]:


year = 2024
month = 3


# In[24]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[ ]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[ ]:


y_pred.std()


# In[ ]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[ ]:


df['preds'] = y_pred


# In[ ]:


df.head()


# In[ ]:


df_result = df[['ride_id','preds']]
df_result.head()


# In[ ]:


output_file = '/workspaces/mlops/04-deployment/df_result.pqt'


# In[ ]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[ ]:




