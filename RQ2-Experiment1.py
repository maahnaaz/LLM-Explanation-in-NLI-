# In this experiment we have 1 step
# step 1: s1,s2 --> exp + label
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import transformers
import pandas as pd
from huggingface_hub import login
import bitsandbytes
import subprocess
import json

from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer



json_schema1 = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"}
    }
}


json_schema2 = {
    "type": "object",
    "properties": {
        "label": {"type": "string"}
    }
}


# Load e_snli original data
df_train_1 = pd.read_csv("../Datasets/eSNLI/esnli_train_1.csv")
df_train_2 = pd.read_csv("../Datasets/eSNLI/esnli_train_2.csv")
df_train = pd.concat([df_train_1, df_train_2])
df_dev = pd.read_csv("../Datasets/eSNLI/esnli_dev.csv")
df_test = pd.read_csv("../Datasets/eSNLI/esnli_test.csv")

model_name = "meta-llama/Llama-2-70b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# If you want to load the model explicity
quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16
    )

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", quantization_config=quant_config) 


pipeline = transformers.pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

with open("../RQ2-exp1.json", "w") as file:
    batch = 0
    for i in range(8488, len(df_dev[:9842])):
        if i%500==0: 
            print(i)
        prompt1 = f"""
 [INST] <<SYS>>
<persona>
You are an expert in English language syntax and semantics.
You are knowledgeable in classifying the semantic relationship between pairs of sentences.
</persona>
<goal>
Your goal is to craft an explanation to describe the relationship between two given sentences. 
</goal>
<</SYS>>
<instruction>
Think step by step
1 - You are given two sentences.
2 - You are provided with a label indicating the relationship of the two given sentences, which could be 'entailment', 'contradiction', or 'neutral'.
3-  Select some keywords from sentences based on the label.
4 - If the label is a contradiction, choose at least one keyword from each sentence. 
5 - If the label is entailment, choose at least one keyword from the Premise. 
6 - If the label is neutral, only choose keywords from the Hypothesis. 
7 - Use the chosen keywords and craft an explanation sentence.
8 - The explanation sentence should justify why the label accurately represents the relationship between the sentences.
9 - The explanation sentence must contain between 3 to 35 tokens.
10 - Reflect on your answers before outputting it.
</instruction>
<constraints>
Consider the given label and selected keywords when crafting the explanation sentence.
The explanation sentence MUST contain at least 3 and at most 35 tokens. 
</constraints>
<format>
Output the results in JSON format.
</format>
<example>
<example of contradiction>
input:    
 Premise: "A person on a horse jumps over a broken-down airplane.",
 Hypothesis: "A person is at a diner, ordering an omelet."
 label: "Contradiction"
Premise_keywords: ["on", "a", "horse", "jumps"],
Hypothesis_keywords: ["is", "at", "a", "diner", "ordering"],
explanation: "One cannot be on a jumping horse and cannot be a diner ordering food."
<example of neutral>
input:
 Premise: "A person on a horse jumps over a broken down airplane.",
 Hypothesis: "A person is training his horse for a competition."
 label: "Neutral"
Premise_keywords: [],
Hypothesis_keywords: ["training", "his", "horse"],
explanation: "The person is not necessarily training his horse."
<example of entailment>
input: 
 Premise: "A person on a horse jumps over a broken down airplane.",
 Hypothesis: "A person is outdoors, on a horse."
 label: "Entailment"
Premise_keywords: ["a", "broken", "down", "airplane"],
Hypothesis_keywords: ["outdoors", "on", "a", "horse"],
explanation: "A broken down airplane is outdoors."
</example>
input:  
  "Premise": '{df_dev.iloc[i]['Sentence1']}',
  "Hypothesis": '{df_dev.iloc[i]['Sentence2']}'
  "label": '{df_dev.iloc[i]['gold_label']}'
Premise_keywords: 
Hypothesis_keywords:
explanation:
 [/INST]
         """


        jsonformer = Jsonformer(model, tokenizer, json_schema1, prompt1, max_number_tokens=6, max_string_token_length=35)

        generated_data1 = jsonformer()
        
        
        
        prompt2 = f"""

[INST] <<SYS>>
<persona>
You are an expert in English language syntax and semantics.
You are knowledgeable in classifying the semantic relationship between pairs of sentences.
</persona>
<goal>
Your goal is to classify the relationship between two given sentences. The possible relationships are 'entailment or 'contradiction' or 'neutral'. 
</goal>
<</SYS>>
<instruction>
Think step by step
1 - You are given two sentences.
2 - You are provided with an explanation sentence describing the relationship of the two given sentences.
3 - You assign a label for the sentence pair, from the 3 possible labels that are 'entailment or 'contradiction' or 'neutral'. 
4 - Assign 'entailment' when Sentence1 logically implies and leads to Sentence2. 
5 - Assign 'contradiction' when Sentence1 and Sentence2 can not be true at the same time.
6 - Assign 'neutral' when you cannot find any sufficiently strong reasons to satisfy either entailment or contradiction.
7 - Reflect on your answers before outputting it.
</instruction>
<constraints>
Your label output can only be ONE of 3 values: either 'entailment' or 'contradiction' or 'neutral'.
When assigning the label, consider both the explanation sentence and the pairs of sentences.
</constraints>
<format>
Output the results in JSON format.
</format>
<example>
<example of contradiction>
input:    
 Premise: "A person on a horse jumps over a broken-down airplane.",
 Hypothesis: "A person is at a diner, ordering an omelet."
 explanation: "One cannot be on a jumping horse and cannot be a diner ordering food."
label: "Contradiction"
<example of neutral>
input:
 Premise: "A person on a horse jumps over a broken down airplane.",
 Hypothesis: "A person is training his horse for a competition."
 explanation: "The person is not necessarily training his horse."
label: "Neutral"
<example of entailment>
input: 
 Premise: "A person on a horse jumps over a broken down airplane.",
 Hypothesis: "A person is outdoors, on a horse."
 explanation: "A broken down airplane is outdoors."
label: "Entailment"
</example>
input:  
 "Premise": '{df_dev.iloc[i]['Sentence1']}',
 "Hypothesis": '{df_dev.iloc[i]['Sentence2']}',
 "explanation": '{generated_data1['explanation']}'

label:
 [/INST]
        """


        #print(prompt)
        jsonformer1 = Jsonformer(model, tokenizer, json_schema2, prompt2, max_number_tokens=600, max_string_token_length=35)

        generated_data2 = jsonformer1()
        
        generated_data1['ID'] = str(df_dev.iloc[i]['pairID'])
        generated_data1['label_p2'] = str(generated_data2['label'])

        file.write(json.dumps(generated_data1,indent=2))
        file.flush()

file.close()




