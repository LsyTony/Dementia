import torch
from torch.utils.data.dataset import Dataset
import json
import pickle
from sklearn.model_selection import train_test_split
import os

class TransformerYelpDataset(Dataset):
    def __init__(self, tokenizer, data, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        note = []
        medicalnotes, label = self.data[index]['medicalnotes'], self.data[index]["label"]#list of dictionary
        study_ids=self.data[index]['patient_id']
        #print('dataset_get_item_method:','medicalnotes:',medicalnotes,'label',label)
        medicalnotes = " ".join(medicalnotes) #add empty space between medicalnotes
        #print('dataset_get_item_method after:','medicalnotes:',medicalnotes,'label',label)
        #scores = self.data[index]['scores']
        return medicalnotes, label, study_ids
      
    def collate_fn(self, data):
        #List of sentences and frames [B,]
        inputs, label, study_ids = zip(*data)
        #print('inputs',inputs,'inputs type',type(inputs),'label',label,'avg score type',type(label),)
        #print('study id',study_ids,'type',type(study_ids))
        encodings = self.tokenizer.batch_encode_plus(
            list(inputs),
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            #return_token_type_ids=False, # Longformer does not make use of token type ids, therefore a list of zeros is returned.
            return_attention_mask=True,
            return_tensors='pt'#return pytorch tensors
        ) #[B, max_len in this batch] tokenize a batch of sequences
        
        inputs_text=self.tokenizer.batch_decode(encodings['input_ids'],skip_special_tokens=True )
        score = torch.LongTensor(label)#dont know num of row but knows num of column
        #study_ids= torch.LongTensor(tuple(int(x) for x in study_ids))
        if 'token_type_ids' not in encodings.keys():
            encodings['token_type_ids'] = torch.zeros((1,))
        return encodings['input_ids'], encodings['attention_mask'], encodings['token_type_ids'], score, study_ids, inputs_text

    def __len__(self):
        return len(self.data)
