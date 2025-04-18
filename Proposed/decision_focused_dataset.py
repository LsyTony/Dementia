import torch
from torch.utils.data.dataset import Dataset
import json
import pickle
from sklearn.model_selection import train_test_split
import os
import random
import nltk
from nltk.tokenize import sent_tokenize
from decision_focused_selection import DecisionFocusedSentenceSelector
nltk.download('punkt')


class DecisionFocusedDataset(Dataset):
    def __init__(self, tokenizer, data, max_len, 
                 selector=None, iteration=1, max_iterations=10):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Initialize sentence selector if not provided
        if selector is None:
            self.selector = DecisionFocusedSentenceSelector(
                max_tokens=max_len,
                relevance_threshold=1.0,
                iteration=iteration,
                max_iterations=max_iterations
            )
        else:
            self.selector = selector

    def __getitem__(self, index):
        # Extract patient data
        patient_id = self.data[index].get('patient_id', str(index))
        original_text = self.data[index].get('notes', '')
        reviews = self.data[index].get('reviews', [])
        
        # Handle different data formats
        if original_text:
            text_to_process = original_text
        elif reviews:
            if isinstance(reviews, list):
                text_to_process = " ".join(reviews)
            else:
                text_to_process = reviews
        else:
            text_to_process = ""
        
        # Get label
        label = self.data[index].get('avg_score', self.data[index].get('label', 0))
        
        # Select sentences based on decision-focused approach
        selected_text = self.selector.select_sentences(
            text_to_process, 
            patient_id,
            self.tokenizer
        )
        return selected_text, label, patient_id

    def collate_fn(self, data):
        #List of sentences and frames [B,]
        inputs, labels, patient_ids = zip(*data)
        #print('collate_fn_inputs',inputs,'inputs type',type(inputs),'avg_score',avg_score,'avg score type',type(avg_score))
        #print('collate_fn select index',selectindex_list,'select index type', type(selectindex_list))
       # Tokenize inputs
        encodings = self.tokenizer.batch_encode_plus(
            list(inputs),
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors='pt'
        )
        # Convert labels to tensor
        labels_tensor = torch.LongTensor(labels)
        
        # Handle token_type_ids for different model types
        if 'token_type_ids' not in encodings.keys():
            encodings['token_type_ids'] = torch.zeros((1,))
            
        return (
            encodings['input_ids'], 
            encodings['attention_mask'], 
            encodings['token_type_ids'], 
            labels_tensor, 
            inputs, 
            list(patient_ids)
        )      

    def __len__(self):
        return len(self.data)

