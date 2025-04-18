import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import logging
import os

logger = logging.getLogger(__name__)

class DecisionFocusedInference:
    def __init__(self, 
                 model_path, 
                 max_tokens=4096, 
                 sentence_encoder=None):
        
        self.model_path = model_path
        self.max_tokens = max_tokens
        
        # Initialize sentence encoder
        if sentence_encoder is None:
            self.sentence_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        else:
            self.sentence_encoder = sentence_encoder
    
    def load_centroids_from_csv(self, csv_path):
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Separate the ID column (assuming it's the first column)
        id_column = df.iloc[:, 0]
        
        # Extract the embedding values (all columns except the ID)
        embeddings = df.iloc[:, 1:].values
        
        logger.info(f"Loaded {len(embeddings)} centroids with dimension {embeddings.shape[1]} from {csv_path}")
        
        return embeddings, id_column.values
    
    def load_centroids(self, case_centroids_path, control_centroids_path):
        # Load case and control centroids
        case_embeddings, case_ids = self.load_centroids_from_csv(case_centroids_path)
        control_embeddings, control_ids = self.load_centroids_from_csv(control_centroids_path)
        
        return {
            'case_embeddings': case_embeddings,
            'case_ids': case_ids,
            'control_embeddings': control_embeddings,
            'control_ids': control_ids
        }
    
    def select_sentences_for_inference(self, document, tokenizer, case_centroids_path, control_centroids_path):
        # Load centroids if not already done
        if not hasattr(self, 'centroids'):
            self.centroids = self.load_centroids(case_centroids_path, control_centroids_path)
            
        # Split document into sentences
        sentences = sent_tokenize(document)
        
        # Get embeddings for sentences
        sentence_embeddings = self.sentence_encoder.encode(sentences)
        
        # Combine case and control centroid embeddings
        all_centroid_embeddings = np.vstack([
            self.centroids['case_embeddings'],
            self.centroids['control_embeddings']
        ])
        
        # Compute similarity matrix between sentences and centroids
        similarity_matrix = cosine_similarity(sentence_embeddings, all_centroid_embeddings)
        
        # Select sentences based on highest similarity to centroids
        # while maintaining textual non-redundancy
        selected = []
        selected_indices = []
        token_count = 0
        
        # For each sentence, compute its max similarity to any centroid
        # and its max similarity to any already selected sentence
        while token_count < self.max_tokens and len(selected_indices) < len(sentences):
            remaining_indices = [i for i in range(len(sentences)) if i not in selected_indices]
            if not remaining_indices:
                break
                
            # For each remaining sentence, calculate:
            # 1. Sum of max similarities to centroids
            # 2. Sum of max similarities to already selected sentences (for non-redundancy)
            best_score = -float('inf')
            best_idx = -1
            
            for idx in remaining_indices:
                # Calculate centroid similarity (objective to maximize)
                centroid_sim = np.sum(np.max(similarity_matrix[idx]))
                
                # Calculate redundancy (objective to minimize)
                if not selected_indices:
                    redundancy = 0
                else:
                    redundancy = np.sum([cosine_similarity([sentence_embeddings[idx]], 
                                                           [sentence_embeddings[j]])[0][0] 
                                         for j in selected_indices])
                
                # Combined score (maximize similarity, minimize redundancy)
                score = centroid_sim - redundancy
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            # Add selected sentence if it doesn't exceed token limit
            tokens = len(tokenizer.encode(sentences[best_idx]))
            if token_count + tokens <= self.max_tokens:
                selected.append(sentences[best_idx])
                selected_indices.append(best_idx)
                token_count += tokens
            else:
                break
        
        return " ".join(selected)


def process_and_save_selected_content(
    model_path,
    case_centroids_path,
    control_centroids_path,
    medical_notes_file,
    output_csv="test_data.csv",
    max_tokens=4096
):
    
    # Initialize the inference utility
    inference = DecisionFocusedInference(
        model_path=model_path,
        max_tokens=max_tokens
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("paraphrase-MiniLM-L6-v2")
    
    # Load medical notes data
    if os.path.exists(medical_notes_file):
        if medical_notes_file.endswith('.csv'):
            notes_data = pd.read_csv(medical_notes_file)
        elif medical_notes_file.endswith('.json'):
            notes_data = pd.read_json(medical_notes_file)
        else:
            # Assume it's a text file with one patient per line
            with open(medical_notes_file, 'r') as f:
                notes_text = f.readlines()
            
            # Create dataframe
            notes_data = pd.DataFrame({
                'patient_id': [f"patient_{i}" for i in range(len(notes_text))],
                'notes': notes_text
            })
    else:
        raise Exception("Medical notes file not exist")
    
    # Process each patient's notes
    results = []
    for _, row in notes_data.iterrows():
        patient_id = row['patient_id']
        notes = row['notes']
        
        # Select relevant sentences
        selected_content = inference.select_sentences_for_inference(
            document=notes,
            tokenizer=tokenizer,
            case_centroids_path=case_centroids_path,
            control_centroids_path=control_centroids_path
        )
        
        results.append({
            'patient_id': patient_id,
            'original_notes': notes,
            'selected_content': selected_content,
            'original_token_count': len(tokenizer.encode(notes)),
            'selected_token_count': len(tokenizer.encode(selected_content))
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Saved selected content for {len(results_df)} patients to {output_csv}")
    
    return results_df


if __name__ == "__main__":
    # Example execution with test paths - modify these to your actual paths
    model_path = "./models/dementia_model"
    case_centroids_path = "./data/case_centroids.csv"
    control_centroids_path = "./data/control_centroids.csv"
    medical_notes_file = "./data/test_medical_notes.csv"
    
    # Process notes and save selected content
    result_df = process_and_save_selected_content(
        model_path=model_path,
        case_centroids_path=case_centroids_path,
        control_centroids_path=control_centroids_path,
        medical_notes_file=medical_notes_file,
        output_csv="test_data.csv"
    )
    
    # Display summary
    print("\nSummary of processing:")
    print(f"Total patients processed: {len(result_df)}")