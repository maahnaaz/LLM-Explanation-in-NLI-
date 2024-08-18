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

with open("../RQ1-exp2.json", "w") as file:
    batch = 0
    for i in range(8488, len(df_dev[:9842])):
        if i%500==0: 
            print(i)
        prompt1 = f"""
        [INST] <<SYS>>
Your goal is to describe the relationship between the given sentences.
Explain any differences between the situations described in the given pairs of sentences.
Explain any similarities between the given pairs of sentences.
<</SYS>>
Given sentences: 
  "Premise": '{df_dev.iloc[i]['Sentence1']}',
  "Hypothesis": '{df_dev.iloc[i]['Sentence2']}'
"explanation": 
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
 explanation: "The premise describes an outdoor scenario. In contrast, the hypothesis situates a person indoors at a diner."
label: "Contradiction"
<example of neutral>
input:
 Premise: "A person on a horse jumps over a broken down airplane.",
 Hypothesis: "A person is training his horse for a competition."
 explanation: "Both sentences involve horses but do not directly support or contradict each other."
label: "Neutral"
<example of entailment>
input: 
 Premise: "A person on a horse jumps over a broken down airplane.",
 Hypothesis: "A person is outdoors, on a horse."
 explanation: "The premise states that a person on a horse jumps over an airplane, clearly indicating that the person is outdoors."
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





