import torch
from nltk import bigrams
from collections import Counter
import json
import nltk
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import evaluate

bertscore = evaluate.load("bertscore")
rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")


df_esnli = pd.read_csv('../eSNLI/esnli_dev.csv')

choice = 2 # [0, 1, 2]
df_llm = None
if choice == 0:
    df_llm0 = pd.read_csv('../results/eval_llm/eval05.csv')
    df_llm1 = pd.read_csv('../results/eval_llm/eval010.csv')
    df_llm = pd.concat([df_llm0,df_llm1], ignore_index=True)
    # Dropping duplicate IDs but keeping the first occurrence
    df_llm = df_llm.drop_duplicates(subset='ID', keep='first')
    
if choice == 1:
    df_llm = pd.read_csv('../results/expthenlabel/expthenlabel02.csv')
    df_esnli = df_esnli.iloc[:len(df_llm)]
    
if choice == 2:
    df_llm0 = pd.read_csv('../results/cot_llm/cot_exp04.csv')
    df_llm1 = pd.read_csv('../results/cot_llm/cot_exp57.csv')
    df_llm = pd.concat([df_llm0,df_llm1], ignore_index=True)
    
    
    
    

predicted_labels = df_llm['label']
true_labels = df_esnli['gold_label']
report = classification_report(true_labels, predicted_labels)

with open("codes/analysis_results.txt", "a") as f:
    f.write("classification report") 
    f.write(f"{report}\n")
f.close()
   
# Extract the 'explanation' column
explanations_eslni = df_esnli['Explanation_1'].astype(str)
explanations_llm = df_llm['explanation'].astype(str)
# Bertscore
results_b = bertscore.compute(predictions=explanations_eslni.values.tolist(), 
                            references=explanations_llm.values.tolist(), 
                            model_type="distilbert-base-uncased")
with open("codes/analysis_results.txt", "a") as f:
    f.write("\nBertScore\n")
    print("\x1b[1;35mBertScore:\x1b[0m")
    for key in ['precision', 'recall', 'f1']:
        print(f"{key}: mean={np.mean(results_b[key])} & std={np.std(results_b[key])}")
        f.write(f"{key}: mean={np.mean(results_b[key])} & std={np.std(results_b[key])}\n")
f.close()
# ROUGE
results_r = rouge.compute(predictions=explanations_eslni.values.tolist(), 
                            references=explanations_llm.values.tolist())
with open("codes/analysis_results.txt", "a") as f:
    print("\x1b[1;36mROUGE:\x1b[0m")
    print(results_r)
    f.write("\nROUGE\n")
    f.write(json.dumps(results_r)+"\n")
f.close()
# BLUE
results_bl = bleu.compute(predictions=explanations_eslni.values.tolist(), 
                            references=explanations_llm.values.tolist())
with open("codes/analysis_results.txt", "a") as f:
    f.write("\nBLEU\n")
    print("\x1b[1;34mBLEU:\x1b[0m")
    for key in results_bl.keys():
        if key == 'precisions':
            print(f"{key}: mean={np.mean(results_bl[key])} & std={np.std(results_bl[key])}")
            text = f"{key}: mean={np.mean(results_bl[key])} & std={np.std(results_bl[key])}"
        else:
            print(f"{key}: {results_bl[key]}")
            text = f"{key}: {results_bl[key]}"
        f.write(text+"\n")
f.close()
