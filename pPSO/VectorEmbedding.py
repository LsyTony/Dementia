#!/usr/bin/env python
# coding: utf-8

# In[56]:


import sys
# ~ get_ipython().system('{sys.executable} -m pip install emoji')
# ~ get_ipython().system(' pip install transformers')
# ~ get_ipython().system(' pip install --upgrade sentence-transformers==1.2.1    # THE NEWEST VERSION BREAKS WITH BERT BASE ON OUR IMPLEMENTATION')
# ~ get_ipython().system(' pip install umap-learn')


# In[62]:


import os
import umap
import re
import pandas as pd
import torch as tc
import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm

from sklearn import preprocessing

# Set the Scaler to normalize between [0-1]
scaler = preprocessing.MinMaxScaler()

# Set the relevant UMAP input parameters
reduced_num_dimensions = 300    # Number of dimensions to which reduce embedded vectors
global_struct_capture_var = 35    # Value to capture the global structure in UMAP reduction

# Specify which embedding algorithm to use (Default ~ stsb-bert-base)
embedding = "stsb-bert-base"    # roberta-base, bert-base-uncased, longformer-base-4096, stsb-bert-base

# Input processed text data file
infile = "./case.csv"

# Output full-size and reduced embedding files
outfile_full = "./case_embedding.csv"
outfile_reduced = "./reduced_embedding.csv"

do_reduce = False          # Toggle for enabling/disabling dimension reduction
output_full_embd = True    # Toggle for enabling/disabling the output of full-size embeddings


# In[58]:


# load model with RoBERTa Embedding
if embedding == 'roberta-base':
    from transformers import RobertaTokenizer, RobertaModel
    tokenizer = RobertaTokenizer.from_pretrained(str(embedding))
    modelEmb = RobertaModel.from_pretrained("roberta-base")

# load model with BERT Embedding
elif embedding == 'bert-base-uncased':
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    modelEmb = BertModel.from_pretrained("bert-base-uncased")

# load model with Longformer Embedding
elif embedding == 'longformer-base-4096':
    from transformers import LongformerTokenizer, LongformerModel
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    modelEmb = LongformerModel.from_pretrained("allenai/longformer-base-4096")

# load model with AugSBERT Embedding
elif embedding == 'stsb-bert-base':
    from sentence_transformers import SentenceTransformer
    #modelEmb = SentenceTransformer('models/sbert_models/stsb-bert-base')
    modelEmb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# In[59]:


# Takes in the full size embeddings and returns a reduced version to the specified dimensionality using UMAP
def reduce_dimensions(full_dim_df, num_dims=300, g_var=35):
    print("\nReducing...")
    # NOTE: Can sometimes take a while to run, depends on number of dimensions and avaliable resources
    
    reducer = umap.UMAP(n_neighbors=int(g_var), n_components=int(num_dims), metric='cosine', random_state=42)
    scaler = preprocessing.MinMaxScaler()

    # Split the input into the metadata & embedded vector for each document
    metadf = full_dim_df[full_dim_df.columns[0:3]]
    embdf = full_dim_df[full_dim_df.columns[3:]]

    tempdf = scaler.fit_transform(embdf.to_numpy())    
    r_embd = reducer.fit_transform(tempdf)  
    r_embd = scaler.fit_transform(r_embd)

    r_cols = list(range(num_dims))

    r_dfo = pd.DataFrame(columns=r_cols)

    for j in range(int(r_embd.shape[0])):
        r_dfo.loc[j] = r_embd[j]
    
        if (j % 100 == 0):    # This is what shows the progress
            print(j, end=" ")

    r_outdf = pd.concat([metadf, r_dfo], axis=1)

    return r_outdf


# In[63]:


def convert_to_embedding(indf):
    embed = list()
    empty_rows = list()  # For removal by index (in case of rows with empty text fields)
    num_rows = 0
    
    num_docs = int(indf.shape[0])
    print(num_docs,"num_docs")
    print("\nEmbedding...")
    
    for i, row in indf.iterrows():
        num_rows += 1
        tempin = str(row[0])
        #print(tempin)
        if len(tempin.split()) > 0:
            if embedding == 'stsb-bert-base':
                tempin = tempin[:512]    # Max BERT Size
                sent_toks = (modelEmb.tokenize(tempin))['input_ids'][0].tolist()

                encoded_dict = modelEmb.encode(tempin, output_value='token_embeddings', convert_to_tensor=True)
                #print("dict shape",encoded_dict.shape)
            elif embedding == 'longformer-base-4096':
                encoded_dict = tokenizer.encode(tempin, truncation=True, max_length=4096)    # For longformer only

            else:        
                encoded_dict = tokenizer.encode(tempin, truncation=True, max_length=512)

            if (i % 100 == 0):    # This is what shows the progress
                print(i, end=" ")

            encoded_tensor = tc.tensor(encoded_dict).unsqueeze(0)
            #print("tensor shape",encoded_tensor.shape)
            #encoded_tensor = tc.tensor(encoded_dict)
            
            if embedding == 'stsb-bert-base':
                outputs = encoded_tensor

            else:
                outputs = modelEmb(encoded_tensor)

            last_hidden_states_embeddings = outputs[0].squeeze(0) 
            #print("last shape",last_hidden_states_embeddings.shape)
            doc_embedding = tc.mean(last_hidden_states_embeddings, dim=0)
            #print("doc shape",doc_embedding.shape)
            embed.append(doc_embedding)

        # Removal of rows with empty docs
        else:
            empty_rows.apppend(int(i))
    
    cols = list(range(int(embed[0].numpy().shape[0])))
    output_df = indf.drop(labels=empty_rows, axis=0)
    tempdf = pd.DataFrame(columns=cols)

    for j in range(num_rows):        
        tempdf.loc[j] = embed[j].numpy() #tensor -> numpy()
    
    n_embd = scaler.fit_transform(tempdf.to_numpy())    #dataframe -> to_numpy()
    dfo = pd.DataFrame(data=n_embd, columns=cols)

    # "inner" is chosen to ensure there are always valuse avaliable for clustering
    output_df = pd.concat([output_df, dfo], axis=1, join="inner")

    return output_df


# In[66]:


indf = pd.read_csv(str(infile))
indf.head(10)


# In[67]:


with tc.no_grad():    
# Generate full length embedding
    outdf_full = convert_to_embedding(indf)

if output_full_embd:    # Output the full length embedding
    outdf_full.to_csv(outfile_full, index=False, encoding="utf-8")

if do_reduce:    # Reduce the embeddings
    outdf_reduced = reduce_dimensions(outdf_full, reduced_num_dimensions, global_struct_capture_var)

    # Output the dimensionally reduced embeddings
    outdf_reduced.to_csv(outfile_reduced, index=False, encoding="utf-8")


# In[65]:


# ~ infile = "./control.csv"

# ~ # Output full-size and reduced embedding files
# ~ outfile_full = "./control_embedding.csv"
# ~ indf = pd.read_csv(str(infile))
# ~ indf.head(10)



# ~ # In[ ]:


# ~ with tc.no_grad():    
# ~ # Generate full length embedding
    # ~ outdf_full = convert_to_embedding(indf)

# ~ if output_full_embd:    # Output the full length embedding
    # ~ outdf_full.to_csv(outfile_full, index=False, encoding="utf-8")

# ~ if do_reduce:    # Reduce the embeddings
    # ~ outdf_reduced = reduce_dimensions(outdf_full, reduced_num_dimensions, global_struct_capture_var)

    # ~ # Output the dimensionally reduced embeddings
    # ~ outdf_reduced.to_csv(outfile_reduced, index=False, encoding="utf-8")
