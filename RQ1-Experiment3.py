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
        "explanation_contradiction": {"type": "string"},
        "explanation_neutral": {"type": "string"},
        "explanation_entailment": {"type": "string"}
    }
}


json_schema2 = {
    "type": "object",
    "properties": {
        "label_contradiction": {"type": "string"}
    }
}

json_schema3 = {
    "type": "object",
    "properties": {
        "label_neutral": {"type": "string"}
    }
}
json_schema4 = {
    "type": "object",
    "properties": {
        "label_entailment": {"type": "string"}
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

with open("../RQ1-exp3.json", "w") as file:
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


        jsonformer1 = Jsonformer(model, tokenizer, json_schema1, prompt1, max_number_tokens=6, max_string_token_length=35)

        generated_data1 = jsonformer1()
        
        
        
        prompt2 = f"""
[INST] <<SYS>>
<persona>
You are an expert in English language syntax and semantics.
You are knowledgeable in classifying the semantic relationship between pairs of sentences.
</persona>
<goal>
Your goal is to classify the relationship between two given sentences. The possible relationships are 'entailment or 'contradiction' or 'neutral' based on a given explanation.
</goal>
<</SYS>>
<instruction>
Think step by step
1 - You are given two sentences.
2 - You are provided with an explanation sentence describing the relationship of the two given sentences.
3 - You assign a label for the sentence pair, from the three possible labels that are 'entailment or 'contradiction' or 'neutral' based on the given explanation. 
5 - If the explanation states that the sentences describe 2 different situations or activities, assign the contradiction label.
6 - If the sentences can not be happening at the same time, assign the contradiction label.
7 - If the explanations state that sentences are similar, assign entailment.
8 - If the Premise logically leads to the Hypothesis, assign entailment.
9 - If the explanations show that one sentence is an instance of the other one, assign entailment. 
10 - If the Hypothesis is an instance of Premise, assign entailment. 
11 - If you cannot find any sufficiently strong reasons to satisfy either entailment or contradiction, the sentences are neutral.
12 - If the explanation states the sentences are both similar and different, the sentences are neutral.
</instruction>
<format>
output the results in JSON format.
</format>
<example>
<example of contradiction>
 Premise: "Under a blue sky with white clouds, a child reaches up to touch the propeller of a plane standing parked on a field of grass.",
 Hypothesis: "A child is playing with a ball."
 explanation: "Sentence1 describes a specific and elaborate scene with a child interacting with a plane, while Sentence2 describes a general and simple scenario of a child playing with a ball. Sentence1 is a descriptive sentence, while Sentence2 is a declarative sentence.",
label: "Contradiction"
<example of neutral>
 Premise: "A man selling donuts to a customer during a world exhibition event held in the city of Angeles",
 Hypothesis: "A man selling donuts to a customer during a world exhibition event while people wait in line behind him.",
 explanation: "There is a difference between the two sentences. Sentence 1 describes a man selling donuts to a customer during a world exhibition event held in the city of Angeles, while sentence 2 describes a man selling donuts to a customer during a world exhibition event while people wait in line behind him. The main difference is that sentence 2 provides additional information about people waiting in line behind the customer being sold donuts, which is not present in sentence 1."
label: "Neutral"
<example of entailment>
 Premise: "A man selling donuts to a customer during a world exhibition event held in the city of Angeles",
 Hypothesis: "A man selling donuts to a customer."
 explanation: "The two sentences describe the same scenario, i.e., a man selling donuts to a customer. However, sentence 1 provides additional information, specifying the location as the city of Angeles during a world exhibition event. Sentence 2 does not provide such additional information and is more general."
label: "Entailment"
</example>
 "Premise": '{df_dev.iloc[i]['Sentence1']}',
 "Hypothesis": '{df_dev.iloc[i]['Sentence2']}',
 "explanation": '{generated_data1['explanation_contradiction']}'
label:
 [/INST]
        """
        jsonformer2 = Jsonformer(model, tokenizer, json_schema2, prompt2, max_number_tokens=600, max_string_token_length=35)
        generated_data2 = jsonformer2()
        prompt3 = f"""
[INST] <<SYS>>
<persona>
You are an expert in English language syntax and semantics.
You are knowledgeable in classifying the semantic relationship between pairs of sentences.
</persona>
<goal>
Your goal is to classify the relationship between two given sentences. The possible relationships are 'entailment or 'contradiction' or 'neutral' based on a given explanation.
</goal>
<</SYS>>
<instruction>
Think step by step
1 - You are given two sentences.
2 - You are provided with an explanation sentence describing the relationship of the two given sentences.
3 - You assign a label for the sentence pair, from the three possible labels that are 'entailment or 'contradiction' or 'neutral' based on the given explanation. 
5 - If the explanation states that the sentences describe 2 different situations or activities, assign the contradiction label.
6 - If the sentences can not be happening at the same time, assign the contradiction label.
7 - If the explanations state that sentences are similar, assign entailment.
8 - If the Premise logically leads to the Hypothesis, assign entailment.
9 - If the explanations show that one sentence is an instance of the other one, assign entailment. 
10 - If the Hypothesis is an instance of Premise, assign entailment. 
11 - If you cannot find any sufficiently strong reasons to satisfy either entailment or contradiction, the sentences are neutral.
12 - If the explanation states the sentences are both similar and different, the sentences are neutral.
</instruction>
<format>
output the results in JSON format.
</format>
<example>
<example of contradiction>
 Premise: "Under a blue sky with white clouds, a child reaches up to touch the propeller of a plane standing parked on a field of grass.",
 Hypothesis: "A child is playing with a ball."
 explanation: "Sentence1 describes a specific and elaborate scene with a child interacting with a plane, while Sentence2 describes a general and simple scenario of a child playing with a ball. Sentence1 is a descriptive sentence, while Sentence2 is a declarative sentence.",
label: "Contradiction"
<example of neutral>
 Premise: "A man selling donuts to a customer during a world exhibition event held in the city of Angeles",
 Hypothesis: "A man selling donuts to a customer during a world exhibition event while people wait in line behind him.",
 explanation: "There is a difference between the two sentences. Sentence 1 describes a man selling donuts to a customer during a world exhibition event held in the city of Angeles, while sentence 2 describes a man selling donuts to a customer during a world exhibition event while people wait in line behind him. The main difference is that sentence 2 provides additional information about people waiting in line behind the customer being sold donuts, which is not present in sentence 1."
label: "Neutral"
<example of entailment>
 Premise: "A man selling donuts to a customer during a world exhibition event held in the city of Angeles",
 Hypothesis: "A man selling donuts to a customer."
 explanation: "The two sentences describe the same scenario, i.e., a man selling donuts to a customer. However, sentence 1 provides additional information, specifying the location as the city of Angeles during a world exhibition event. Sentence 2 does not provide such additional information and is more general."
label: "Entailment"
</example>
 "Premise": '{df_dev.iloc[i]['Sentence1']}',
 "Hypothesis": '{df_dev.iloc[i]['Sentence2']}',
 "explanation": '{generated_data1['explanation_neutral']}'
label:
 [/INST]
        """
        jsonformer3 = Jsonformer(model, tokenizer, json_schema3, prompt3, max_number_tokens=600, max_string_token_length=35)
        generated_data3 = jsonformer3()
        prompt4 = f"""
[INST] <<SYS>>
<persona>
You are an expert in English language syntax and semantics.
You are knowledgeable in classifying the semantic relationship between pairs of sentences.
</persona>
<goal>
Your goal is to classify the relationship between two given sentences. The possible relationships are 'entailment or 'contradiction' or 'neutral' based on a given explanation.
</goal>
<</SYS>>
<instruction>
Think step by step
1 - You are given two sentences.
2 - You are provided with an explanation sentence describing the relationship of the two given sentences.
3 - You assign a label for the sentence pair, from the three possible labels that are 'entailment or 'contradiction' or 'neutral' based on the given explanation. 
5 - If the explanation states that the sentences describe 2 different situations or activities, assign the contradiction label.
6 - If the sentences can not be happening at the same time, assign the contradiction label.
7 - If the explanations state that sentences are similar, assign entailment.
8 - If the Premise logically leads to the Hypothesis, assign entailment.
9 - If the explanations show that one sentence is an instance of the other one, assign entailment. 
10 - If the Hypothesis is an instance of Premise, assign entailment. 
11 - If you cannot find any sufficiently strong reasons to satisfy either entailment or contradiction, the sentences are neutral.
12 - If the explanation states the sentences are both similar and different, the sentences are neutral.
</instruction>
<format>
output the results in JSON format.
</format>
<example>
<example of contradiction>
 Premise: "Under a blue sky with white clouds, a child reaches up to touch the propeller of a plane standing parked on a field of grass.",
 Hypothesis: "A child is playing with a ball."
 explanation: "Sentence1 describes a specific and elaborate scene with a child interacting with a plane, while Sentence2 describes a general and simple scenario of a child playing with a ball. Sentence1 is a descriptive sentence, while Sentence2 is a declarative sentence.",
label: "Contradiction"
<example of neutral>
 Premise: "A man selling donuts to a customer during a world exhibition event held in the city of Angeles",
 Hypothesis: "A man selling donuts to a customer during a world exhibition event while people wait in line behind him.",
 explanation: "There is a difference between the two sentences. Sentence 1 describes a man selling donuts to a customer during a world exhibition event held in the city of Angeles, while sentence 2 describes a man selling donuts to a customer during a world exhibition event while people wait in line behind him. The main difference is that sentence 2 provides additional information about people waiting in line behind the customer being sold donuts, which is not present in sentence 1."
label: "Neutral"
<example of entailment>
 Premise: "A man selling donuts to a customer during a world exhibition event held in the city of Angeles",
 Hypothesis: "A man selling donuts to a customer."
 explanation: "The two sentences describe the same scenario, i.e., a man selling donuts to a customer. However, sentence 1 provides additional information, specifying the location as the city of Angeles during a world exhibition event. Sentence 2 does not provide such additional information and is more general."
label: "Entailment"
</example>
 "Premise": '{df_dev.iloc[i]['Sentence1']}',
 "Hypothesis": '{df_dev.iloc[i]['Sentence2']}',
 "explanation": '{generated_data1['explanation_entailment']}'
label:
 [/INST]
        """

        jsonformer4 = Jsonformer(model, tokenizer, json_schema4, prompt4, max_number_tokens=600, max_string_token_length=35)

        generated_data4 = jsonformer4()
        
        generated_data1['ID'] = str(df_dev.iloc[i]['pairID'])
        generated_data1['label_contradition'] = str(generated_data2['label'])
        generated_data1['label_neutral'] = str(generated_data3['label'])
        generated_data1['label_entailment'] = str(generated_data4['label'])
        file.write(json.dumps(generated_data1,indent=2))
        file.flush()

file.close()





