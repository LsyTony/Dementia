import torch
import random
import numpy as np
import nltk
import csv
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import math

class DecisionFocusedSentenceSelector:
    def __init__(self, 
                 max_tokens=4096, 
                 relevance_threshold=1.0, 
                 iteration=1, 
                 max_iterations=10,
                 simulated_annealing_lambda=0.1,
                 sentence_encoder=None):
        self.max_tokens = max_tokens
        self.relevance_threshold = relevance_threshold
        self.iteration = iteration
        self.max_iterations = max_iterations
        self.simulated_annealing_lambda = simulated_annealing_lambda
        self.sentence_relevance_scores = {}  # Dictionary to store relevance scores
        self.selected_sentences = {}  # Dictionary to track selected sentences for each patient
        
        # Initialize sentence encoder for similarity calculations
        if sentence_encoder is None:
            self.sentence_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        else:
            self.sentence_encoder = sentence_encoder
    
    def split_into_sentences(self, document):
        
        return sent_tokenize(document)
    
    def get_sentence_embeddings(self, sentences):
        
        return self.sentence_encoder.encode(sentences)
    
    def compute_similarity_matrix(self, sentences):
        
        embeddings = self.get_sentence_embeddings(sentences)
        return cosine_similarity(embeddings)
    
    def calculate_epsilon(self):
        
        return (self.max_iterations - self.iteration + 1) / self.max_iterations
    
    def select_sentences(self, document, patient_id, tokenizer):
        
        sentences = self.split_into_sentences(document)
        
        # If first iteration, select sentences randomly
        if self.iteration == 1:
            selected = []
            token_count = 0
            
            # Randomly shuffle sentences and select until max_tokens is reached
            random_order = list(range(len(sentences)))
            random.shuffle(random_order)
            
            for idx in random_order:
                tokens = len(tokenizer.encode(sentences[idx]))
                if token_count + tokens <= self.max_tokens:
                    selected.append(sentences[idx])
                    token_count += tokens
                if token_count >= self.max_tokens:
                    break
                    
            # Store selected sentences for this patient
            self.selected_sentences[patient_id] = selected
            return " ".join(selected)
        
        # For later iterations, use epsilon-greedy approach with relevance scores
        else:
            # Compute similarity matrix for textual non-redundancy
            similarity_matrix = self.compute_similarity_matrix(sentences)
            
            selected = []
            selected_indices = []
            token_count = 0
            
            # Use epsilon-greedy approach
            epsilon = self.calculate_epsilon()
            
            while token_count < self.max_tokens and len(selected) < len(sentences):
                # Decide whether to explore or exploit
                if random.random() < epsilon:  # Explore - random selection
                    remaining_indices = [i for i in range(len(sentences)) if i not in selected_indices]
                    if not remaining_indices:
                        break
                    
                    next_idx = random.choice(remaining_indices)
                else:  # Exploit - select based on relevance scores
                    # Filter sentences with relevance score >= threshold
                    qualified_indices = []
                    for i in range(len(sentences)):
                        if i not in selected_indices:
                            sentence_key = f"{patient_id}_{i}"
                            if sentence_key in self.sentence_relevance_scores and \
                               self.sentence_relevance_scores[sentence_key] >= self.relevance_threshold:
                                qualified_indices.append(i)
                    
                    if not qualified_indices:
                        # If no qualified sentences, fall back to exploration
                        remaining_indices = [i for i in range(len(sentences)) if i not in selected_indices]
                        if not remaining_indices:
                            break
                        next_idx = random.choice(remaining_indices)
                    else:
                        # Select the sentence that minimizes redundancy per Equation 1 in the paper
                        min_redundancy = float('inf')
                        next_idx = -1
                        
                        for idx in qualified_indices:
                            # Calculate sum of maximum similarities to previously selected sentences
                            if not selected_indices:  # If first sentence
                                redundancy = 0
                            else:
                                max_similarities = [similarity_matrix[idx][j] for j in selected_indices]
                                redundancy = sum(max_similarities)
                            
                            if redundancy < min_redundancy:
                                min_redundancy = redundancy
                                next_idx = idx
                
                # Add selected sentence if it doesn't exceed token limit
                tokens = len(tokenizer.encode(sentences[next_idx]))
                if token_count + tokens <= self.max_tokens:
                    selected.append(sentences[next_idx])
                    selected_indices.append(next_idx)
                    token_count += tokens
                else:
                    break
            
            # Store selected sentences for this patient
            self.selected_sentences[patient_id] = [sentences[i] for i in selected_indices]
            return " ".join(selected)
    
    def update_relevance_scores(self, patient_id, prediction_correct):
        
        if patient_id not in self.selected_sentences:
            return
            
        sentences = self.selected_sentences[patient_id]
        
        for i, _ in enumerate(sentences):
            sentence_key = f"{patient_id}_{i}"
            
            # Initialize if not exists
            if sentence_key not in self.sentence_relevance_scores:
                self.sentence_relevance_scores[sentence_key] = 0
            
            # Update based on Equation 2 in the paper
            q_value = 1 if prediction_correct else -1
            
            # Simplified update rule for relevance score
            current_score = self.sentence_relevance_scores[sentence_key]
            # Moving average update
            self.sentence_relevance_scores[sentence_key] = (current_score + q_value) / 2
    
    def adjust_threshold(self, model, data_loader, num_epochs=5):
        
        current_threshold = self.relevance_threshold
        current_loss = self.evaluate_threshold(model, data_loader, current_threshold)
        
        for k in range(num_epochs):
            # Propose new threshold
            proposed_threshold = 1 - 2 * k / num_epochs
            
            # Evaluate loss with proposed threshold
            proposed_loss = self.evaluate_threshold(model, data_loader, proposed_threshold)
            
            # Calculate delta loss
            delta = current_loss - proposed_loss
            
            # Simulated annealing acceptance probability
            accept_probability = 1.0 if delta >= 0 else math.exp(
                delta / (self.simulated_annealing_lambda * (1 - self.iteration / self.max_iterations + 0.01))
            )
            
            # Accept or reject proposed threshold
            if random.random() < accept_probability:
                current_threshold = proposed_threshold
                current_loss = proposed_loss
        
        self.relevance_threshold = current_threshold
        return current_threshold
    
    def evaluate_threshold(self, model, data_loader, threshold):
        
        # Store original threshold value
        original_threshold = self.relevance_threshold
        
        # Set new threshold temporarily
        self.relevance_threshold = threshold
        
        # Track losses
        total_loss = 0.0
        total_samples = 0
        
        # Put model in evaluation mode
        model.eval()
        
        with torch.no_grad():
            for batch in data_loader:
                # Process the batch with the model
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                
                # Add token_type_ids if needed by the model
                if hasattr(model, 'config') and model.config.model_type in ["bert", "xlnet", "albert"]:
                    inputs["token_type_ids"] = batch[2]
                
                # Get model outputs
                outputs = model(**inputs)
                loss = outputs[0]
                
                # Accumulate loss
                batch_size = batch[0].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Put model back in training mode
        model.train()
        
        # Restore original threshold
        self.relevance_threshold = original_threshold
        
        # Calculate average loss
        average_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        return average_loss
    
    def export_selected_sentences_to_csv(self, output_path):
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['patient_id', 'sentence_idx', 'sentence', 'relevance_score'])
            
            for patient_id, sentences in self.selected_sentences.items():
                for i, sentence in enumerate(sentences):
                    sentence_key = f"{patient_id}_{i}"
                    score = self.sentence_relevance_scores.get(sentence_key, 0)
                    writer.writerow([patient_id, i, sentence, score])
    
    def next_iteration(self):
       
        self.iteration += 1
        if self.iteration > self.max_iterations:
            self.iteration = self.max_iterations