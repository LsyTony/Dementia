import pandas as pd
import numpy as np
import torch
from tqdm.notebook import tqdm

raw_dataset_case= pd.read_csv("/geode2/projects/iu/IN-REGI-PDM/Dementia/Dementia-Workspace/Shengyang/processed_data/-2.5year_to_-0.5year/merge_case_-2.5y_to_-0.5y.csv")
raw_dataset_control= pd.read_csv("/geode2/projects/iu/IN-REGI-PDM/Dementia/Dementia-Workspace/Shengyang/processed_data/-2.5year_to_-0.5year/merge_control_-2.5y_to_-0.5y_matched.csv")
raw_dataset_case=raw_dataset_case.iloc[:,1:]
raw_dataset_control=raw_dataset_control.iloc[:,1:]

from transformers import AutoTokenizer
from textsum.summarize import Summarizer

#model_name = "pszemraj/led-base-book-summary"
model_name = "patrickvonplaten/led-large-16384-pubmed"
tokenizer_out = AutoTokenizer.from_pretrained(model_name)
summarizer = Summarizer(
    model_name_or_path=model_name,  # you can use any Seq2Seq model on the Hub
    is_general_attention_model= False, #whether the model is a general attention model(bert) or not a general model(LED), defaults to True
    token_batch_length=16384,  # how many tokens to batch summarize at a time, refers to max input supported length to tokenizer
    max_length_ratio = 1/16, #the ratio of the token_batch_length to use as the max_length for the model generation output(text) 
    repetition_penalty = 1.5, #The parameter for repetition penalty. 1.0 means no penalty. higher means more penalty
    no_repeat_ngram_size=3, #If set to int > 0, all ngrams of that size can only occur once.
    encoder_no_repeat_ngram_size= 0, #If set to int > 0, all ngrams of that size that occur in the encoder_input_ids cannot occur in the decoder_input_ids.
    length_penalty = 1.0, #Exponential penalty to the length that is used with beam-based generation.
    early_stopping = True, #Controls the stopping condition for beam-based methods, like beam-search. 
    num_beams =4, #Number of beams for beam search. 1 means no beam search.
    num_beam_groups=1, #Number of groups to divide num_beams into in order to ensure diversity among different groups of beams.
)
# ~ long_string = "This is a long string of text that will be summarized."
# ~ out_str = summarizer.summarize_string(long_string)
# ~ print(f"summary: {out_str}")

print("finish loading model")

summary_dataset_case=raw_dataset_case.copy(deep=True)
print(summary_dataset_case.equals(raw_dataset_case))

summary_dataset_control=raw_dataset_control.copy(deep=True)
print(summary_dataset_control.equals(raw_dataset_control))

def ToSummary(summary_dataset: pd.DataFrame, mark:str) -> pd.DataFrame:
    for i,row in enumerate(summary_dataset.itertuples()):
        text_to_summarize=str(row[2])
        #print(text_to_summarize)
        tokenized_sentences = tokenizer_out.tokenize(text_to_summarize)
        #print(len(tokenized_sentences))
        summarizer.inference_params['min_length']=min(int(len(tokenized_sentences)*0.75),1024)
        summarizer.inference_params['max_length']=min(len(tokenized_sentences),1024)
        #print(summarizer.get_inference_params())
        summary = summarizer.summarize_string(text_to_summarize)
        #print(len(tokenizer_out.tokenize(summary)))
        if i%10==0:
            print("processing row index at",i)
        #print("original text",text_to_summarize,"\n")
        #print("summary",summary)
        summary_dataset.loc[i,"REP"]=summary
        if i%200==0:
            if mark == "case":
                summary_dataset.to_csv("/geode2/projects/iu/IN-REGI-PDM/Dementia/Dementia-Workspace/Shengyang/processed_data/-2.5year_to_-0.5year/abstractive_summary_pubmed_case_-2.5y_to_-0.5y.csv")
            elif mark == "control":
                summary_dataset.to_csv("/geode2/projects/iu/IN-REGI-PDM/Dementia/Dementia-Workspace/Shengyang/processed_data/-2.5year_to_-0.5year/abstractive_summary_pubmed_control_-2.5y_to_-0.5y.csv")
    return summary_dataset
    
# ~ def ToSummary(summary_dataset: pd.DataFrame,summarizer) -> pd.DataFrame:
    # ~ for i,row in enumerate(summary_dataset.itertuples()):
        # ~ #if i < 19680:
            # ~ #continue
          
        # ~ if i == 10796 or i == 18429 or i == 19684:
            # ~ summarizer = Summarizer(
                # ~ model_name_or_path=model_name,  # you can use any Seq2Seq model on the Hub
                # ~ is_general_attention_model= False, #whether the model is a general attention model(bert) or not a general model(LED), defaults to True
                # ~ token_batch_length=16384,  # how many tokens to batch summarize at a time, refers to max input supported length
                # ~ max_length_ratio = 0.25, #the ratio of the token_batch_length to use as the max_length for the model generation output(text) 
                # ~ repetition_penalty = 2.5, #The parameter for repetition penalty. 1.0 means no penalty. higher means more penalty
            # ~ )
            # ~ print("index:",i,"load 2.5 repetition penalty")
        
        # ~ text_to_summarize=str(row[2])
        # ~ summary = summarizer.summarize_string(text_to_summarize)
        
        # ~ if i == 10796 or i == 18429 or i == 19684:
            # ~ summarizer = Summarizer(
                # ~ model_name_or_path=model_name,  # you can use any Seq2Seq model on the Hub
                # ~ is_general_attention_model= False, #whether the model is a general attention model(bert) or not a general model(LED), defaults to True
                # ~ token_batch_length=16384,  # how many tokens to batch summarize at a time, refers to max input supported length
                # ~ max_length_ratio = 0.25, #the ratio of the token_batch_length to use as the max_length for the model generation output(text) 
                # ~ repetition_penalty = 1.5, #The parameter for repetition penalty. 1.0 means no penalty. higher means more penalty
            # ~ )
            # ~ print("index:",i,"load 1.5 repetition penalty")
           			
        # ~ if i%10==0:
            # ~ print("processing row index at",i)
        # ~ #print("original text",text_to_summarize,"\n")
        # ~ #print("summary",summary)
        # ~ summary_dataset.loc[i,"REP"]=summary				
    # ~ return summary_dataset

#test version, finding the note that causes cuda error    
# ~ def ToSummary(summary_dataset: pd.DataFrame) -> pd.DataFrame:
    # ~ for i,row in enumerate(summary_dataset.itertuples()):
        # ~ if i%10==0:
            # ~ print("processing row index at",i)
        # ~ if i>6300:
            # ~ text_to_summarize=str(row[2])
            # ~ summary = summarizer.summarize_string(text_to_summarize)            
        # ~ #print("original text",text_to_summarize,"\n")
        # ~ #print("summary",summary)
            # ~ #summary_dataset.loc[i,"REP"]=summary
    # ~ return summary_dataset
    
print("begin processing case")
summary_dataset_case=ToSummary(summary_dataset_case,"case")
print(summary_dataset_case.equals(raw_dataset_case))

summary_dataset_case.to_csv("/geode2/projects/iu/IN-REGI-PDM/Dementia/Dementia-Workspace/Shengyang/processed_data/-2.5year_to_-0.5year/abstractive_summary_pubmed_case_-2.5y_to_-0.5y.csv")

print("begin processing control")
summary_dataset_control=ToSummary(summary_dataset_control,"control")
print(summary_dataset_control.equals(raw_dataset_control))

summary_dataset_control.to_csv("/geode2/projects/iu/IN-REGI-PDM/Dementia/Dementia-Workspace/Shengyang/processed_data/-2.5year_to_-0.5year/abstractive_summary_pubmed_control_-2.5y_to_-0.5y.csv")
