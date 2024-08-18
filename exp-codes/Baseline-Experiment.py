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
        "label": {"type": "string"},
        "Premise_Keywords": {
            "type": "array",
            "items": {"type": "string"}
        },
        "Hypothesis_Keywords": {
            "type": "array",
            "items": {"type": "string"}
        },
        "explanation": {"type": "string"}
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

with open("./results/eval_llm/eval10.json", "w") as file:
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
<</SYS>>
<goal>
Your goal is to classify the relationship between two given sentences and create an explanation to justify your choice. The possible relationships are 'entailment or 'contradiction' or 'neutral'.
</goal>
<instruction>
Think step by step
1 - You are given a premise hypothesis sentence pair.
2 - You assign a label for the sentence pair, from the three possible labels that are 'entailment or 'contradiction' or 'neutral'. 
3 - Assign 'entailment' when the Premise logically implies and leads to the Hypothesis. 
4 - Assign 'contradiction' when the Premise and Hypothesis can not be true simultaneously.
5 - Assign 'neutral' when you cannot find sufficiently strong reasons to satisfy either entailment or contradiction.
6- Select some keywords from sentences based on the label.
7 - If the label is a contradiction, choose at least one keyword from each sentence. 
8 - If the label is entailment, choose at least one keyword from Premise. 
9 - If the label is neutral, only choose keywords from Hypothesis. 
10 - Use the chosen keywords and craft an explanation sentence.
11 - The explanation sentence should justify why the label accurately represents the sentence's relationship.
12 - Reflect on your answers before outputting it.
</instruction>
<constraints>
Your label output can only be ONE of 3 values: either 'entailment' or 'contradiction' or 'neutral'.
The explanation sentence must contain between 3 to 35 tokens.
</constraints>
<example>
<example of contradiction>
 Premise: "A person on a horse jumps over a broken-down airplane.",
 Hypothesis: "A person is at a diner, ordering an omelet."
 
label: "Contradiction",
Premise_keywords: ["on", "a", "horse", "a"],
Hypothesis_keywords: ["is", "at", "a", "diner", "ordering"],
explanation: "One cannot be on a jumping horse and cannot be a diner ordering food."
<example of neutral>
 Premise: "A person on a horse jumps over a broken down airplane.",
 Hypothesis: "A person is training his horse for a competition."
label: "Neutral",
Premise_keywords: [],
Hypothesis_keywords: ["training", "his", "horse"],
explanation: "The person is not necessarily training his horse."
<example of entailment> 
 Premise: "A person on a horse jumps over a broken down airplane.",
 Hypothesis: "A person is outdoors, on a horse."
label: "Entailment",
Premise_keywords: ["a", "broken", "down", "airplane"],
Hypothesis_keywords: ["outdoors", "on", "a", "horse"],
explanation: "A broken down airplane is outdoors."
</example>
<format>
Output the results in JSON format.
</format>

 "Premise": '{df_dev.iloc[i]['Sentence1']}',
 "Hypothesis":  '{df_dev.iloc[i]['Sentence2']}'

label:
Premise_Keywords:
Hypothesis_Keywords:
explanation: 

 [/INST]

        """


        #print(prompt)
        jsonformer = Jsonformer(model, tokenizer, json_schema1, prompt1, max_number_tokens=6, max_string_token_length=34)

        generated_data1 = jsonformer()
        
        
        generated_data1['ID'] = str(df_dev.iloc[i]['pairID'])
        file.write(json.dumps(generated_data1,indent=2))
        file.flush()

file.close()



